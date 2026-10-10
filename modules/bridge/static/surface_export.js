(function (global) {
  "use strict";

  // Builds the deflected top-of-deck TIN for export.
  //
  // The deck DTM's triangles can run 50+ ft along the bridge, far longer than
  // the deflection curve stays straight, so only moving the DTM's own vertices
  // would lose the sag between them. Instead each DTM triangle is cut by a
  // regular plan grid: every piece keeps the DTM triangle's plane (so crowns
  // and breaklines are untouched) and gains vertices every `cellSize` ft,
  // where the deflection is sampled. Pieces share their cut points, so the
  // result is a conforming TIN.
  //
  // With an overhang offset, a strip is then added along the deck's side
  // edges, out to the screed line, carrying the deck's cross slope (or held
  // level) and the deflection there. The edge of deck is a break line: the
  // strip is hung from the deck's own edge vertices, so no triangle spans it,
  // and it is also written as LandXML breaklines for software that rebuilds
  // the TIN.

  const MERGE_TOLERANCE = 1e-6;
  const MERGE_CELL = 1e-4;
  // DTM vertices this close (ft) to a grid line are moved onto it.
  const GRID_SNAP = 1e-4;
  // Fraction of a cell the grid is shifted off the DTM's minimum corner.
  const GRID_SHIFT = 0.381966;

  function signedArea(a, b, c) {
    return ((b.e - a.e) * (c.n - a.n) - (c.e - a.e) * (b.n - a.n)) / 2;
  }

  /** Vertex store that merges points closer than MERGE_TOLERANCE. */
  function vertexStore() {
    const vertices = [];
    const buckets = new Map();
    const keyOf = (e, n) => `${Math.floor(e / MERGE_CELL)}|${Math.floor(n / MERGE_CELL)}`;

    return {
      vertices,
      /** Index of the vertex at (e, n); `make` supplies its data if it is new. */
      add(e, n, make) {
        const ce = Math.floor(e / MERGE_CELL);
        const cn = Math.floor(n / MERGE_CELL);
        for (let de = -1; de <= 1; de += 1) {
          for (let dn = -1; dn <= 1; dn += 1) {
            const bucket = buckets.get(`${ce + de}|${cn + dn}`);
            if (!bucket) continue;
            for (let i = 0; i < bucket.length; i += 1) {
              const vertex = vertices[bucket[i]];
              if (Math.abs(vertex.e - e) <= MERGE_TOLERANCE && Math.abs(vertex.n - n) <= MERGE_TOLERANCE) {
                return bucket[i];
              }
            }
          }
        }
        const index = vertices.length;
        vertices.push({ e, n, ...make() });
        const key = keyOf(e, n);
        if (!buckets.has(key)) buckets.set(key, []);
        buckets.get(key).push(index);
        return index;
      },
    };
  }

  /** Sutherland-Hodgman clip of a convex polygon to one axis-aligned half-plane. */
  function clip(polygon, axis, value, keepGreater) {
    const output = [];
    const inside = (p) => (keepGreater ? p[axis] >= value : p[axis] <= value);
    for (let i = 0; i < polygon.length; i += 1) {
      const current = polygon[i];
      const previous = polygon[(i + polygon.length - 1) % polygon.length];
      const currentIn = inside(current);
      const previousIn = inside(previous);
      if (currentIn !== previousIn) {
        const t = (value - previous[axis]) / (current[axis] - previous[axis]);
        const point = {
          e: previous.e + t * (current.e - previous.e),
          n: previous.n + t * (current.n - previous.n),
        };
        point[axis] = value; // land exactly on the grid line
        output.push(point);
      }
      if (currentIn) output.push(current);
    }
    return output;
  }

  function dropRepeats(polygon) {
    const output = [];
    polygon.forEach((point) => {
      const last = output[output.length - 1];
      if (!last || Math.abs(last.e - point.e) > MERGE_TOLERANCE || Math.abs(last.n - point.n) > MERGE_TOLERANCE) {
        output.push(point);
      }
    });
    while (
      output.length > 1 &&
      Math.abs(output[0].e - output[output.length - 1].e) <= MERGE_TOLERANCE &&
      Math.abs(output[0].n - output[output.length - 1].n) <= MERGE_TOLERANCE
    ) {
      output.pop();
    }
    return output;
  }

  /**
   * Triangulates a convex polygon while keeping every boundary point (the
   * neighbouring pieces rely on them). A plain fan works unless the apex is
   * in line with a run of boundary points; then fan from the centroid.
   */
  function triangulateConvex(polygon) {
    const minArea = 1e-9;
    const fan = [];
    let fanOk = true;
    for (let i = 1; i + 1 < polygon.length; i += 1) {
      if (Math.abs(signedArea(polygon[0], polygon[i], polygon[i + 1])) < minArea) {
        fanOk = false;
        break;
      }
      fan.push([0, i, i + 1]);
    }
    if (fanOk) return { centroid: null, triangles: fan };

    const centroid = polygon.reduce((sum, p) => ({ e: sum.e + p.e, n: sum.n + p.n }), { e: 0, n: 0 });
    centroid.e /= polygon.length;
    centroid.n /= polygon.length;
    const triangles = [];
    for (let i = 0; i < polygon.length; i += 1) {
      const j = (i + 1) % polygon.length;
      if (Math.abs(signedArea(centroid, polygon[i], polygon[j])) >= minArea) triangles.push([-1, i, j]);
    }
    return { centroid, triangles };
  }

  /**
   * @param {object} input
   *   points, faces        the deck DTM (points {e, n, z}, faces [a, b, c])
   *   isopachAt(e, n)      deflection (ft) to add at a plan position
   *   cellSize             grid spacing (ft) for densifying the deck
   *   overhangOffset       ft beyond the deck edge (perpendicular to it) to extend to, or null
   *   fascias              exterior girder lines [{fascia, outward}], to find the deck side edges
   *   deckZ(e, n)          undeflected deck elevation, for the cross slope
   *   crossSlopeRun        ft of deck, inward from the edge, that sets the cross slope
   *   overhangSlope        "slope" (default) carries the cross slope out; "level" holds the edge elevation
   */
  function buildDeflectedSurface(input) {
    const { points, faces, isopachAt, cellSize, fascias, deckZ } = input;
    const overhangOffset = input.overhangOffset;
    const crossSlopeRun = input.crossSlopeRun ?? 2;
    const overhangSlope = input.overhangSlope === "level" ? "level" : "slope";
    const store = vertexStore();
    const out = [];

    let originE = Infinity;
    let originN = Infinity;
    points.forEach((p) => {
      if (p.e < originE) originE = p.e;
      if (p.n < originN) originN = p.n;
    });
    // Start the grid an odd fraction of a cell off the DTM, so grid lines do
    // not systematically land on the model's own (often round) coordinates.
    originE -= cellSize * GRID_SHIFT;
    originN -= cellSize * GRID_SHIFT;

    // A DTM vertex a hair off a grid line would leave a sliver gap between
    // the pieces cut on either side of it; put such vertices on the line.
    // Every triangle sharing the vertex sees the same snapped position.
    const snap = (value, origin) => {
      const line = origin + Math.round((value - origin) / cellSize) * cellSize;
      return Math.abs(value - line) <= GRID_SNAP ? line : value;
    };
    const snapped = points.map((p) => (p ? { e: snap(p.e, originE), n: snap(p.n, originN), z: p.z } : p));

    faces.forEach((face) => {
      let [a, b, c] = face.map((index) => snapped[index]);
      if (!a || !b || !c) return;
      const area = signedArea(a, b, c);
      if (Math.abs(area) < 1e-12) return;
      if (area < 0) [b, c] = [c, b];

      // The DTM triangle's plane, used for every piece cut from it.
      const planeZ = (e, n) => {
        const w1 = signedArea({ e, n }, b, c) / signedArea(a, b, c);
        const w2 = signedArea(a, { e, n }, c) / signedArea(a, b, c);
        return w1 * a.z + w2 * b.z + (1 - w1 - w2) * c.z;
      };
      const vertexData = (e, n) => () => {
        const z = planeZ(e, n);
        return { deckZ: z, z: z + isopachAt(e, n) };
      };

      const minE = Math.min(a.e, b.e, c.e);
      const maxE = Math.max(a.e, b.e, c.e);
      const minN = Math.min(a.n, b.n, c.n);
      const maxN = Math.max(a.n, b.n, c.n);
      const i0 = Math.floor((minE - originE) / cellSize);
      const i1 = Math.floor((maxE - originE) / cellSize);
      const j0 = Math.floor((minN - originN) / cellSize);
      const j1 = Math.floor((maxN - originN) / cellSize);

      for (let i = i0; i <= i1; i += 1) {
        const x0 = originE + i * cellSize;
        const x1 = x0 + cellSize;
        for (let j = j0; j <= j1; j += 1) {
          const y0 = originN + j * cellSize;
          const y1 = y0 + cellSize;

          let polygon = [
            { e: a.e, n: a.n },
            { e: b.e, n: b.n },
            { e: c.e, n: c.n },
          ];
          polygon = clip(polygon, "e", x0, true);
          if (polygon.length) polygon = clip(polygon, "e", x1, false);
          if (polygon.length) polygon = clip(polygon, "n", y0, true);
          if (polygon.length) polygon = clip(polygon, "n", y1, false);
          polygon = dropRepeats(polygon);
          if (polygon.length < 3) continue;

          const { centroid, triangles } = triangulateConvex(polygon);
          const ids = polygon.map((p) => store.add(p.e, p.n, vertexData(p.e, p.n)));
          const centroidId = centroid ? store.add(centroid.e, centroid.n, vertexData(centroid.e, centroid.n)) : null;
          triangles.forEach(([p, q, r]) => {
            out.push([p < 0 ? centroidId : ids[p], ids[q], ids[r]]);
          });
        }
      }
    });

    const deckFaces = out.length;
    let stripFaces = 0;
    let sideEdges = 0;
    let breaklines = [];
    if (overhangOffset !== null && overhangOffset !== undefined && overhangOffset > 0) {
      ({
        added: stripFaces,
        sideEdges,
        breaklines,
      } = addScreedStrip(store, out, { overhangOffset, fascias, deckZ, crossSlopeRun, overhangSlope }));
    }

    // Consistent counter-clockwise faces.
    const vertices = store.vertices;
    const faceList = out
      .filter(([p, q, r]) => p !== q && q !== r && p !== r)
      .map(([p, q, r]) => (signedArea(vertices[p], vertices[q], vertices[r]) < 0 ? [p, r, q] : [p, q, r]));

    return { vertices, faces: faceList, deckFaces, stripFaces, sideEdges, breaklines };
  }

  /**
   * Adds the strip from the deck's side edges out to the screed line, which
   * lies `overhangOffset` beyond the deck edge, perpendicular to it. Returns
   * the number of faces added and the deck-edge break lines (vertex index
   * chains).
   */
  function addScreedStrip(store, out, options) {
    const { overhangOffset, fascias, deckZ, crossSlopeRun, overhangSlope } = options;
    const vertices = store.vertices;
    const sides = deckSideEdges(tinBoundary(vertices, out), fascias);

    // Cross slope at each side-edge vertex, over the last `crossSlopeRun` ft of
    // deck. At a deck corner that run can land on the end edge, where the DTM
    // gives no elevation; take the slope from a neighbour along the edge then.
    // A level overhang has no slope to read.
    const slopes = new Map();
    if (overhangSlope !== "level") {
      sides.vertexNormals.forEach((normal, index) => {
        const v = vertices[index];
        const inner = deckZ(v.e - normal.e * crossSlopeRun, v.n - normal.n * crossSlopeRun);
        if (inner !== null) slopes.set(index, (v.deckZ - inner) / crossSlopeRun);
      });
      for (let pass = 0; pass < 3; pass += 1) {
        sides.edges.forEach((edge) => {
          if (!slopes.has(edge.ia) && slopes.has(edge.ib)) slopes.set(edge.ia, slopes.get(edge.ib));
          if (!slopes.has(edge.ib) && slopes.has(edge.ia)) slopes.set(edge.ib, slopes.get(edge.ia));
        });
      }
    }

    // One screed point per side-edge vertex, straight out from the deck edge.
    const screed = new Map();
    sides.vertexNormals.forEach((normal, index) => {
      const v = vertices[index];
      const slope = slopes.get(index) ?? 0;
      const screedDeckZ = v.deckZ + slope * overhangOffset;
      // The overhang holds the deflection it has at the deck edge (the fascia
      // girder's), already applied to this vertex. Reading it here rather than
      // at the screed keeps it right where a curved screed line bulges past
      // the chorded isopach mesh.
      const isopach = v.z - v.deckZ;
      screed.set(
        index,
        store.add(v.e + normal.e * overhangOffset, v.n + normal.n * overhangOffset, () => ({
          deckZ: screedDeckZ,
          z: screedDeckZ + isopach,
          screed: true,
        })),
      );
    });

    let added = 0;
    sides.edges.forEach((edge) => {
      const su = screed.get(edge.ia);
      const sv = screed.get(edge.ib);
      if (su === undefined || sv === undefined) return;
      out.push([edge.ia, edge.ib, sv], [edge.ia, sv, su]);
      added += 2;
    });
    return { added, sideEdges: sides.edges.length, breaklines: added ? chainEdges(sides.edges) : [] };
  }

  /** Joins edges ({ia, ib}) that share vertices into polylines of vertex indices. */
  function chainEdges(edges) {
    const adjacency = new Map();
    const link = (from, to, k) => {
      if (!adjacency.has(from)) adjacency.set(from, []);
      adjacency.get(from).push({ to, k });
    };
    edges.forEach((edge, k) => {
      link(edge.ia, edge.ib, k);
      link(edge.ib, edge.ia, k);
    });

    const used = new Set();
    const walk = (start) => {
      const line = [start];
      let current = start;
      for (;;) {
        const step = adjacency.get(current).find((s) => !used.has(s.k));
        if (!step) break;
        used.add(step.k);
        current = step.to;
        line.push(current);
      }
      return line;
    };

    const lines = [];
    // Open chains start at their ends (or at junctions); what is left is closed loops.
    adjacency.forEach((steps, index) => {
      if (steps.length === 2) return;
      while (steps.some((s) => !used.has(s.k))) lines.push(walk(index));
    });
    edges.forEach((edge, k) => {
      if (!used.has(k)) lines.push(walk(edge.ia));
    });
    return lines.filter((line) => line.length >= 2);
  }

  // ---------------------------------------------------------------------------
  // Deck side edges
  //
  // The overhang (screed) line is measured perpendicular to the deck edge,
  // which may curve (it usually follows the alignment) while precast girders
  // are straight. These helpers find the deck's side edges -- the TIN
  // boundary running along each exterior girder -- with outward normals.
  // ---------------------------------------------------------------------------

  const SIDE_EDGE_MAX_ANGLE = 30; // degrees between a side edge and its local girder

  /** Boundary edges of a TIN (used by one face), with unit outward normals. */
  function tinBoundary(points, faces) {
    const edges = new Map();
    faces.forEach(([p, q, r]) => {
      [
        [p, q, r],
        [q, r, p],
        [r, p, q],
      ].forEach(([u, v, w]) => {
        const key = u < v ? `${u}|${v}` : `${v}|${u}`;
        const existing = edges.get(key);
        if (existing) existing.count += 1;
        else edges.set(key, { u, v, w, count: 1 });
      });
    });

    const boundary = [];
    edges.forEach((edge) => {
      if (edge.count !== 1) return;
      const a = points[edge.u];
      const b = points[edge.v];
      const length = Math.hypot(b.e - a.e, b.n - a.n);
      if (length < 1e-9) return;
      const dir = { e: (b.e - a.e) / length, n: (b.n - a.n) / length };
      let normal = { e: -dir.n, n: dir.e };
      const opposite = points[edge.w];
      if ((opposite.e - a.e) * normal.e + (opposite.n - a.n) * normal.n > 0) normal = { e: -normal.e, n: -normal.n };
      boundary.push({ ia: edge.u, ib: edge.v, a, b, dir, normal, length });
    });
    return boundary;
  }

  /** Closest point on a polyline, with the interpolated outward normal there. */
  function closestOnFascia(fascia, e, n) {
    let best = null;
    const { fascia: line, outward } = fascia;
    for (let i = 0; i + 1 < line.length; i += 1) {
      const a = line[i];
      const b = line[i + 1];
      const dE = b.e - a.e;
      const dN = b.n - a.n;
      const lengthSq = dE * dE + dN * dN;
      if (lengthSq < 1e-18) continue;
      const u = Math.max(0, Math.min(1, ((e - a.e) * dE + (n - a.n) * dN) / lengthSq));
      const foot = { e: a.e + u * dE, n: a.n + u * dN };
      const distance = Math.hypot(e - foot.e, n - foot.n);
      if (!best || distance < best.distance) {
        const length = Math.sqrt(lengthSq);
        let oe = (1 - u) * outward[i].e + u * outward[i + 1].e;
        let on = (1 - u) * outward[i].n + u * outward[i + 1].n;
        const olength = Math.hypot(oe, on) || 1;
        oe /= olength;
        on /= olength;
        best = { distance, foot, tangent: { e: dE / length, n: dN / length }, outward: { e: oe, n: on } };
      }
    }
    return best;
  }

  /**
   * Boundary edges that are deck sides: within SIDE_EDGE_MAX_ANGLE of the
   * nearest exterior girder's local direction, outside it, and facing away
   * from it. Using each girder's local direction (not one bridge-wide axis)
   * keeps curved decks and spans with different headings covered.
   * Each side edge gains vertex normals (`na`, `nb`) averaged with its
   * neighbours, so offsets follow a curved edge smoothly.
   *
   * @param fascias  [{fascia: [{e, n}], outward: [{e, n}]}] exterior girder lines
   */
  function deckSideEdges(boundary, fascias) {
    const minCos = Math.cos((SIDE_EDGE_MAX_ANGLE * Math.PI) / 180);
    const sides = boundary.filter((edge) => {
      const mid = { e: (edge.a.e + edge.b.e) / 2, n: (edge.a.n + edge.b.n) / 2 };
      let near = null;
      fascias.forEach((fascia) => {
        const hit = closestOnFascia(fascia, mid.e, mid.n);
        if (hit && (!near || hit.distance < near.distance)) near = hit;
      });
      if (!near) return false;
      const along = Math.abs(edge.dir.e * near.tangent.e + edge.dir.n * near.tangent.n);
      const facesOut = edge.normal.e * near.outward.e + edge.normal.n * near.outward.n > 0;
      const outside = (mid.e - near.foot.e) * near.outward.e + (mid.n - near.foot.n) * near.outward.n > 0;
      return along >= minCos && facesOut && outside;
    });

    const sums = new Map();
    sides.forEach((edge) => {
      [edge.ia, edge.ib].forEach((index) => {
        const sum = sums.get(index) || { e: 0, n: 0 };
        sum.e += edge.normal.e;
        sum.n += edge.normal.n;
        sums.set(index, sum);
      });
    });
    const unit = (v) => {
      const length = Math.hypot(v.e, v.n) || 1;
      return { e: v.e / length, n: v.n / length };
    };
    const vertexNormals = new Map();
    sums.forEach((sum, index) => vertexNormals.set(index, unit(sum)));
    sides.forEach((edge) => {
      edge.na = vertexNormals.get(edge.ia);
      edge.nb = vertexNormals.get(edge.ib);
    });
    return { edges: sides, vertexNormals };
  }

  /** Closest point on the deck side edges to (e, n), with the outward normal there. */
  function closestOnDeckEdge(sides, e, n) {
    let best = null;
    sides.edges.forEach((edge) => {
      const { a, b } = edge;
      const dE = b.e - a.e;
      const dN = b.n - a.n;
      const u = Math.max(0, Math.min(1, ((e - a.e) * dE + (n - a.n) * dN) / (edge.length * edge.length)));
      const pe = a.e + u * dE;
      const pn = a.n + u * dN;
      const distance = Math.hypot(e - pe, n - pn);
      if (!best || distance < best.distance) best = { distance, e: pe, n: pn, u, edge };
    });
    if (!best) return null;
    const { edge, u } = best;
    let ne = (1 - u) * edge.na.e + u * edge.nb.e;
    let nn = (1 - u) * edge.na.n + u * edge.nb.n;
    const length = Math.hypot(ne, nn) || 1;
    ne /= length;
    nn /= length;
    return { e: best.e, n: best.n, distance: best.distance, normal: { e: ne, n: nn } };
  }

  function escapeXml(text) {
    return String(text).replace(/[<>&"']/g, (ch) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;", "'": "&apos;" })[ch]);
  }

  /** LandXML 1.2 document with one TIN surface. Points are written N E Z. */
  function toLandXml(surface, options = {}) {
    const name = escapeXml(options.name || "Deflected Top of Deck");
    const desc = escapeXml(options.description || "");
    const units =
      options.unitsXml ||
      '<Units><Imperial areaUnit="squareFoot" linearUnit="USSurveyFoot" volumeUnit="cubicYard" ' +
        'temperatureUnit="fahrenheit" pressureUnit="inchHG" diameterUnit="inch" ' +
        'angularUnit="decimal degrees" directionUnit="decimal degrees"></Imperial></Units>';
    const now = options.now || new Date();
    const pad = (value) => String(value).padStart(2, "0");
    const date = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
    const time = `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;

    let minZ = Infinity;
    let maxZ = -Infinity;
    surface.vertices.forEach((v) => {
      if (v.z < minZ) minZ = v.z;
      if (v.z > maxZ) maxZ = v.z;
    });

    const lines = [
      '<?xml version="1.0" encoding="UTF-8"?>',
      '<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" ' +
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" ' +
        'xsi:schemaLocation="http://www.landxml.org/schema/LandXML-1.2 http://www.landxml.org/schema/LandXML-1.2/LandXML-1.2.xsd" ' +
        `date="${date}" time="${time}" version="1.2" language="English" readOnly="false">`,
      `\t${units}`,
      '\t<Application name="Survey Toolbox" desc="Bridge Superstructure - deflected top of deck" manufacturer="Survey Toolbox"></Application>',
      "\t<Surfaces>",
      `\t\t<Surface name="${name}" desc="${desc}">`,
    ];
    // Break lines (e.g. the edge of deck) as 3D polylines on the TIN's own
    // vertices, so a program that rebuilds the TIN keeps the grade break.
    const breaklines = surface.breaklines ?? [];
    if (breaklines.length) {
      const breaklineName = escapeXml(options.breaklineName || "Edge of deck");
      lines.push("\t\t\t<SourceData>", "\t\t\t\t<Breaklines>");
      breaklines.forEach((line, index) => {
        const coords = line
          .map((i) => surface.vertices[i])
          .map((v) => `${v.n.toFixed(6)} ${v.e.toFixed(6)} ${v.z.toFixed(6)}`)
          .join(" ");
        lines.push(
          `\t\t\t\t\t<Breakline name="${breaklineName} ${index + 1}" brkType="standard">`,
          `\t\t\t\t\t\t<PntList3D>${coords}</PntList3D>`,
          "\t\t\t\t\t</Breakline>",
        );
      });
      lines.push("\t\t\t\t</Breaklines>", "\t\t\t</SourceData>");
    }
    lines.push(
      `\t\t\t<Definition surfType="TIN" elevMax="${maxZ.toFixed(6)}" elevMin="${minZ.toFixed(6)}">`,
      "\t\t\t\t<Pnts>",
    );
    surface.vertices.forEach((v, index) => {
      lines.push(`\t\t\t\t\t<P id="${index + 1}">${v.n.toFixed(6)} ${v.e.toFixed(6)} ${v.z.toFixed(6)}</P>`);
    });
    lines.push("\t\t\t\t</Pnts>", "\t\t\t\t<Faces>");
    surface.faces.forEach(([p, q, r]) => {
      lines.push(`\t\t\t\t\t<F>${p + 1} ${q + 1} ${r + 1}</F>`);
    });
    lines.push("\t\t\t\t</Faces>", "\t\t\t</Definition>", "\t\t</Surface>", "\t</Surfaces>", "</LandXML>", "");
    return lines.join("\n");
  }

  global.BridgeSurfaceExport = {
    buildDeflectedSurface,
    toLandXml,
    tinBoundary,
    deckSideEdges,
    closestOnDeckEdge,
  };
})(typeof window !== "undefined" ? window : globalThis);

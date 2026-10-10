const TEMPLATE_HEADERS = [
  "Span number",
  "Girder number",
  "Girder width (ft)",
  "Girder height (ft)",
  "Camber at midspan (in)",
  "Deflection at midspan (in) [required]",
  "Deflection at quarter span (in) [optional]",
  "Deflection at third span (in) [optional]",
  "Support1 Northing (ft)",
  "Support1 Easting (ft)",
  "Support1 Seat Z (ft)",
  "Bearing height at Support1 (in)",
  "Plate height at Support1 (in)",
  "Support2 Northing (ft)",
  "Support2 Easting (ft)",
  "Support2 Seat Z (ft)",
  "Bearing height at Support2 (in)",
  "Plate height at Support2 (in)",
  "Centerline radius (ft) [optional, 0=straight, +CW, -CCW]",
];

// Two-line on-screen rendering of TEMPLATE_HEADERS: the concept on top,
// unit + required/optional together on the bottom -- keeps columns narrow
// without losing information. Parallel array, same length/order as
// TEMPLATE_HEADERS (which stays unabbreviated for the downloaded template
// workbook). The full "0=straight, +CW, -CCW" note is dropped from the
// on-screen column since it is already covered in the user manual.
const GRID_HEADER_DISPLAY = [
  { concept: "Span number" },
  { concept: "Girder number" },
  { concept: "Girder width", meta: "(ft)" },
  { concept: "Girder height", meta: "(ft)" },
  { concept: "Camber at 1/2 span", meta: "(in)" },
  { concept: "Deflection at 1/2 span", meta: "(in) [required]" },
  { concept: "Deflection at 1/4 span", meta: "(in) [optional]" },
  { concept: "Deflection at 1/3 span", meta: "(in) [optional]" },
  { concept: "Support1 Northing", meta: "(ft)" },
  { concept: "Support1 Easting", meta: "(ft)" },
  { concept: "Support1 Seat Z", meta: "(ft)" },
  { concept: "Bearing height at Support1", meta: "(in)" },
  { concept: "Plate height at Support1", meta: "(in)" },
  { concept: "Support2 Northing", meta: "(ft)" },
  { concept: "Support2 Easting", meta: "(ft)" },
  { concept: "Support2 Seat Z", meta: "(ft)" },
  { concept: "Bearing height at Support2", meta: "(in)" },
  { concept: "Plate height at Support2", meta: "(in)" },
  { concept: "Centerline radius", meta: "(ft) [optional]" },
];

// Zoom window: drag a box corner to corner and the view zooms to exactly that
// box. Plotly's own zoom snaps a thin box to one axis and reshapes the box on
// equal-aspect charts (the plan views), so the window is drawn with a
// select-drag instead and applied in enableZoomWindow.
const ZOOM_WINDOW_BUTTON = {
  name: "zoomWindow",
  title: "Zoom window (drag a diagonal)",
  icon: Plotly.Icons.zoombox,
  attr: "dragmode",
  val: "select",
  click: (gd) => Plotly.relayout(gd, { dragmode: "select" }),
};

// Zoom extents: fit everything drawn. Plotly's reset (home) instead returns to
// the view it first recorded, which on the deck plan may predate the DTM.
const ZOOM_EXTENTS_BUTTON = {
  name: "zoomExtents",
  title: "Zoom extents",
  icon: Plotly.Icons.autoscale,
  click: (gd) => Plotly.relayout(gd, { "xaxis.autorange": true, "yaxis.autorange": true }),
};

// Shared across every Plotly chart so the view tools (zoom window, pan,
// zoom in/out, zoom extents, reset, download) look and behave identically
// everywhere.
const PLOTLY_CONFIG = {
  responsive: true,
  displaylogo: false,
  displayModeBar: true,
  modeBarButtonsToRemove: ["zoom2d", "select2d", "lasso2d", "autoScale2d"],
  modeBarButtonsToAdd: [ZOOM_WINDOW_BUTTON, ZOOM_EXTENTS_BUTTON],
};

// Spread into every chart layout. Plotly draws with its own default font
// unless told otherwise; use the app-wide Inter (annotation and trace text
// inherit it). The view tools sit in a horizontal strip (placed top left by
// styles.css) and the zoom window is the default drag.
const PLOTLY_LAYOUT = {
  font: { family: "Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif" },
  modebar: { orientation: "h" },
  dragmode: "select",
  // "d": the box is always the rectangle on the drag's diagonal. ("any" turns
  // a thin drag into a full-width or full-height band.)
  selectdirection: "d",
  newselection: { line: { color: "#0d6efd", width: 1.5, dash: "dash" } },
  activeselection: { fillcolor: "#0d6efd", opacity: 0.08 },
};

/**
 * Plotly reports clicks through its own emitter (gd.on), not as DOM events,
 * and Plotly.newPlot drops a chart's listeners, so charts re-bind their click
 * handler after every render.
 */
function bindChartClick(gd, handler) {
  gd.removeAllListeners?.("plotly_click");
  gd.on("plotly_click", handler);
}

/**
 * Turns a finished select-drag into a zoom to exactly that box, then clears
 * the selection so no points stay dimmed. Plotly.newPlot drops a chart's
 * event listeners, so this runs after every render.
 */
function enableZoomWindow(gd) {
  // Read the drawn box from the layout's selections rather than the
  // plotly_selected event: that event carries no range when the box holds no
  // data points, which is common when zooming into empty space.
  gd.removeAllListeners?.("plotly_relayout");
  gd.on("plotly_relayout", (update) => {
    const box = update?.selections?.[0];
    if (!box || box.type !== "rect") return;
    const [x0, x1] = [box.x0, box.x1].map(Number).sort((a, b) => a - b);
    const [y0, y1] = [box.y0, box.y1].map(Number).sort((a, b) => a - b);
    // Plotly ignores mouse movement under 8 px on an axis, so a drag flatter
    // (or narrower) than that has no extent there: zoom the other axis only.
    const zoom = { selections: [] };
    if (x1 > x0) zoom["xaxis.range"] = [x0, x1];
    if (y1 > y0) zoom["yaxis.range"] = [y0, y1];
    Plotly.update(gd, { selectedpoints: null }, zoom);
  });
}

const state = {
  sourceRows: [],
  topOfGirderPoints: [],
  profiles: {},
  spanToGirders: {},
  girderGeometry: {},
  logs: [],
  dtm: null,
  dtmTin: null,
  isopachMesh: null,
  deflectedDeck: null,
  alignments: [],
  alignment: null,
  sectionStations: [],
  sectionStation: null,
  // Station range the section box accepts: the bridge's, or the alignment's.
  sectionRange: null,
  sectionRangeNote: "",
  section: null,
  // Bumped whenever the plotted data set changes, so the deck plan keeps the
  // user's zoom while only the section line moves.
  planRevision: 0,
};

const ui = {
  tabDataBtn: document.getElementById("tabDataBtn"),
  tabGraphsBtn: document.getElementById("tabGraphsBtn"),
  tabExportBtn: document.getElementById("tabExportBtn"),
  panelData: document.getElementById("panelData"),
  panelGraphs: document.getElementById("panelGraphs"),
  panelExport: document.getElementById("panelExport"),
  sourceTableHead: document.getElementById("sourceTableHead"),
  sourceTableBody: document.getElementById("sourceTableBody"),
  fileInput: document.getElementById("fileInput"),
  dtmFileInput: document.getElementById("dtmFileInput"),
  uploadStatus: document.getElementById("uploadStatus"),
  dtmUploadStatus: document.getElementById("dtmUploadStatus"),
  intervalsInput: document.getElementById("intervalsInput"),
  progressBar: document.getElementById("progressBar"),
  progressText: document.getElementById("progressText"),
  logOutput: document.getElementById("logOutput"),
  graphSpanSelect: document.getElementById("graphSpanSelect"),
  graphGirderSelect: document.getElementById("graphGirderSelect"),
  profileChart: document.getElementById("profileChart"),
  planChart: document.getElementById("planChart"),
  deckSpanSelect: document.getElementById("deckSpanSelect"),
  deckGirderSelect: document.getElementById("deckGirderSelect"),
  deckChart: document.getElementById("deckChart"),
  deckStatus: document.getElementById("deckStatus"),
  planSurfaceSelect: document.getElementById("planSurfaceSelect"),
  contoursToggle: document.getElementById("contoursToggle"),
  contourIntervalInput: document.getElementById("contourIntervalInput"),
  contourStatus: document.getElementById("contourStatus"),
  overhangInput: document.getElementById("overhangInput"),
  overhangSlopeSelect: document.getElementById("overhangSlopeSelect"),
  alignmentFileInput: document.getElementById("alignmentFileInput"),
  alignmentUploadStatus: document.getElementById("alignmentUploadStatus"),
  alignmentSelect: document.getElementById("alignmentSelect"),
  sectionIntervalInput: document.getElementById("sectionIntervalInput"),
  sectionStationInput: document.getElementById("sectionStationInput"),
  sectionStationToggle: document.getElementById("sectionStationToggle"),
  sectionStationMenu: document.getElementById("sectionStationMenu"),
  sectionStationFeedback: document.getElementById("sectionStationFeedback"),
  sectionPrevBtn: document.getElementById("sectionPrevBtn"),
  sectionNextBtn: document.getElementById("sectionNextBtn"),
  sectionStatus: document.getElementById("sectionStatus"),
  sectionChart: document.getElementById("sectionChart"),
  verticalExaggerationInput: document.getElementById("verticalExaggerationInput"),
};

function setProgress(percent, text) {
  const safe = Math.max(0, Math.min(100, Number(percent) || 0));
  ui.progressBar.style.width = `${safe}%`;
  ui.progressBar.textContent = `${safe}%`;
  ui.progressBar.setAttribute("aria-valuenow", String(safe));
  ui.progressText.textContent = text;
}

function formatSpan(spanRaw) {
  const text = String(spanRaw ?? "").trim();
  return /^\d+$/.test(text) ? text.padStart(2, "0") : text;
}

function formatGirder(girderRaw) {
  const text = String(girderRaw ?? "").trim();
  return /^\d+$/.test(text) ? text.padStart(2, "0") : text;
}

function formatInterval(index) {
  return String(index).padStart(2, "0");
}

function parseNumber(value, fallback = 0) {
  if (value === undefined || value === null || value === "") return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function normalizeAngle(angle) {
  let output = angle;
  while (output <= -Math.PI) output += 2 * Math.PI;
  while (output > Math.PI) output -= 2 * Math.PI;
  return output;
}

function buildCenterlineGeometry(support1E, support1N, support2E, support2N, radiusRaw) {
  const dE = support2E - support1E;
  const dN = support2N - support1N;
  const chordLength = Math.hypot(dE, dN);
  const signedRadius = parseNumber(radiusRaw, 0);

  if (chordLength <= 1e-12 || Math.abs(signedRadius) <= 1e-12) {
    return {
      isCurved: false,
      signedRadius: 0,
      length: chordLength,
      at(t) {
        const centerE = support1E + t * dE;
        const centerN = support1N + t * dN;
        const tangentE = chordLength > 1e-12 ? dE / chordLength : 1;
        const tangentN = chordLength > 1e-12 ? dN / chordLength : 0;
        return { centerE, centerN, tangentE, tangentN, station: chordLength * t };
      },
    };
  }

  const radius = Math.abs(signedRadius);
  const halfChord = chordLength / 2;
  if (radius + 1e-9 < halfChord) {
    throw new Error(`Invalid centerline radius ${signedRadius}: |radius| must be at least ${halfChord.toFixed(3)} ft for this girder length.`);
  }

  const midpointE = (support1E + support2E) / 2;
  const midpointN = (support1N + support2N) / 2;
  const rightUnitE = dN / chordLength;
  const rightUnitN = -dE / chordLength;
  const side = signedRadius > 0 ? 1 : -1;
  const centerOffset = Math.sqrt(Math.max(0, radius * radius - halfChord * halfChord));
  const circleCenterE = midpointE + side * centerOffset * rightUnitE;
  const circleCenterN = midpointN + side * centerOffset * rightUnitN;

  const thetaStart = Math.atan2(support1N - circleCenterN, support1E - circleCenterE);
  const thetaEndRaw = Math.atan2(support2N - circleCenterN, support2E - circleCenterE);
  let delta = normalizeAngle(thetaEndRaw - thetaStart);
  if (side > 0 && delta > 0) delta -= 2 * Math.PI;
  if (side < 0 && delta < 0) delta += 2 * Math.PI;

  const arcLength = Math.abs(delta) * radius;

  return {
    isCurved: true,
    signedRadius,
    length: arcLength,
    at(t) {
      const theta = thetaStart + delta * t;
      const centerE = circleCenterE + radius * Math.cos(theta);
      const centerN = circleCenterN + radius * Math.sin(theta);
      const tangentScale = delta >= 0 ? 1 : -1;
      const tangentE = tangentScale * -Math.sin(theta);
      const tangentN = tangentScale * Math.cos(theta);
      return { centerE, centerN, tangentE, tangentN, station: arcLength * t };
    },
  };
}

function computeParabolaA(observations) {
  if (!observations.length) return 0;
  const basis = (t) => t * t - t;
  if (observations.length === 1) {
    const b = basis(observations[0].t);
    return Math.abs(b) < 1e-12 ? 0 : observations[0].value / b;
  }

  let numerator = 0;
  let denominator = 0;
  observations.forEach((obs) => {
    const b = basis(obs.t);
    numerator += obs.value * b;
    denominator += b * b;
  });

  return Math.abs(denominator) < 1e-12 ? 0 : numerator / denominator;
}

function activateTab(tab) {
  const isData = tab === "data";
  const isGraphs = tab === "graphs";

  ui.panelData.classList.toggle("d-none", !isData);
  ui.panelGraphs.classList.toggle("d-none", !isGraphs);
  ui.panelExport.classList.toggle("d-none", tab !== "export");

  ui.tabDataBtn.classList.toggle("active", isData);
  ui.tabGraphsBtn.classList.toggle("active", isGraphs);
  ui.tabExportBtn.classList.toggle("active", tab === "export");

  if (isGraphs) {
    renderProfileChart();
    renderPlanChart();
  }

  if (tab === "export") {
    refreshSections();
    renderDeflectedDeckChart();
  }
}

function triggerDownload(url, filename) {
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function downloadTemplate() {
  const workbook = XLSX.utils.book_new();
  const sheet = XLSX.utils.aoa_to_sheet([TEMPLATE_HEADERS]);
  XLSX.utils.book_append_sheet(workbook, sheet, "Template");
  const output = XLSX.write(workbook, { bookType: "xlsx", type: "array" });
  const url = URL.createObjectURL(new Blob([output], { type: "application/octet-stream" }));
  triggerDownload(url, "Data_Input Template.xlsx");
}

function readWorkbook(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const bytes = new Uint8Array(event.target.result);
        resolve(XLSX.read(bytes, { type: "array" }));
      } catch (error) {
        reject(error);
      }
    };
    reader.onerror = () => reject(new Error("Failed to read the selected file."));
    reader.readAsArrayBuffer(file);
  });
}

function normalizeRow(row) {
  const result = Array.from({ length: TEMPLATE_HEADERS.length }, (_, i) => row?.[i] ?? "");
  return result;
}

function renderSourceGrid() {
  ui.sourceTableHead.innerHTML = `<tr>${GRID_HEADER_DISPLAY.map(
    ({ concept, meta }) =>
      `<th><span class="grid-th-concept">${concept}</span>${
        meta ? `<span class="grid-th-meta">${meta}</span>` : ""
      }</th>`,
  ).join("")}</tr>`;

  if (!state.sourceRows.length) {
    ui.sourceTableBody.innerHTML = `<tr><td colspan="${TEMPLATE_HEADERS.length}" class="text-center text-secondary py-3">Upload a spreadsheet to view/edit rows.</td></tr>`;
    return;
  }

  ui.sourceTableBody.innerHTML = "";
  state.sourceRows.forEach((row, rowIndex) => {
    const tr = document.createElement("tr");
    TEMPLATE_HEADERS.forEach((_, colIndex) => {
      const td = document.createElement("td");
      const input = document.createElement("input");
      input.className = "grid-cell";
      input.value = row[colIndex] ?? "";
      input.addEventListener("input", (event) => {
        state.sourceRows[rowIndex][colIndex] = event.target.value;
      });
      td.appendChild(input);
      tr.appendChild(td);
    });
    ui.sourceTableBody.appendChild(tr);
  });
}

async function loadSourceRows() {
  const [file] = ui.fileInput.files;
  if (!file) {
    state.sourceRows = [];
    ui.uploadStatus.textContent = "";
    renderSourceGrid();
    return;
  }

  const workbook = await readWorkbook(file);
  const firstSheet = workbook.Sheets[workbook.SheetNames[0]];
  state.sourceRows = XLSX.utils.sheet_to_json(firstSheet, { header: 1 }).slice(1).map(normalizeRow);
  ui.uploadStatus.textContent = "Spreadsheet uploaded correctly. You can edit values in the grid before calculation.";
  renderSourceGrid();
}

function buildGirderPoints(row, intervals) {
  const spanRaw = row[0];
  const girderRaw = row[1];
  const spanDisplay = String(spanRaw ?? "").trim();
  const girderDisplay = String(girderRaw ?? "").trim();

  const girderWidth = parseNumber(row[2]);
  const girderHeight = parseNumber(row[3]);
  const camberMid = parseNumber(row[4]);
  const defMid = parseNumber(row[5], Number.NaN);
  const defQuarter = parseNumber(row[6], Number.NaN);
  const defThird = parseNumber(row[7], Number.NaN);

  const support1N = parseNumber(row[8]);
  const support1E = parseNumber(row[9]);
  const support1Z = parseNumber(row[10]);
  const support1Bearing = parseNumber(row[11]);
  const support1Plate = parseNumber(row[12]);
  const support2N = parseNumber(row[13]);
  const support2E = parseNumber(row[14]);
  const support2Z = parseNumber(row[15]);
  const support2Bearing = parseNumber(row[16]);
  const support2Plate = parseNumber(row[17]);
  const centerlineRadius = parseNumber(row[18], 0);

  if (!spanDisplay || !girderDisplay || !Number.isFinite(defMid)) {
    throw new Error("Span, Girder, and Deflection at midspan are required in each row.");
  }

  const observations = [];
  if (Number.isFinite(defQuarter)) observations.push({ t: 0.25, value: defQuarter });
  if (Number.isFinite(defThird)) observations.push({ t: 1 / 3, value: defThird });
  observations.push({ t: 0.5, value: defMid });

  const aDefIn = computeParabolaA(observations);
  const aCamberIn = -4 * camberMid;

  const bearingFeet1 = support1Bearing / 12;
  const plateFeet1 = support1Plate / 12;
  const bearingFeet2 = support2Bearing / 12;
  const plateFeet2 = support2Plate / 12;

  const centerline = buildCenterlineGeometry(support1E, support1N, support2E, support2N, centerlineRadius);

  const rows = [];
  const graphPoints = [];
  const planCenterline = [];
  const planEdges = [];

  for (let i = 0; i <= intervals; i += 1) {
    const t = i / intervals;
    const geometryPoint = centerline.at(t);
    const centerN = geometryPoint.centerN;
    const centerE = geometryPoint.centerE;

    const seatZ = support1Z + t * (support2Z - support1Z);
    const bearing = bearingFeet1 + t * (bearingFeet2 - bearingFeet1);
    const plate = plateFeet1 + t * (plateFeet2 - plateFeet1);

    const deflectionIn = aDefIn * (t * t - t);
    const camberIn = aCamberIn * (t * t - t);
    const deflectionFt = deflectionIn / 12;
    const camberFt = camberIn / 12;

    const elevation = seatZ + bearing + plate + girderHeight + deflectionFt;

    const halfWidth = girderWidth / 2;
    const perpendicularE = -geometryPoint.tangentN;
    const perpendicularN = geometryPoint.tangentE;
    const leftN = centerN + perpendicularN * halfWidth;
    const leftE = centerE + perpendicularE * halfWidth;
    const rightN = centerN - perpendicularN * halfWidth;
    const rightE = centerE - perpendicularE * halfWidth;

    const base = `${formatSpan(spanRaw)}${formatGirder(girderRaw)}${formatInterval(i)}`;
    rows.push([leftN, leftE, elevation, `${base}L`, deflectionFt, camberFt]);
    rows.push([rightN, rightE, elevation, `${base}R`, deflectionFt, camberFt]);

    graphPoints.push({
      station: geometryPoint.station,
      deflectionIn,
      interval: i,
      t,
    });

    planCenterline.push({ n: centerN, e: centerE });
    // The same left/right edge positions the Top of Girder export uses, so
    // the deflected deck can be reported directly above them.
    planEdges.push({ n: leftN, e: leftE, side: "L" }, { n: rightN, e: rightE, side: "R" });
  }

  return {
    spanDisplay,
    girderDisplay,
    aDefIn,
    aCamberIn,
    support1N,
    support1E,
    support2N,
    support2E,
    centerlineRadius,
    centerlineCurved: centerline.isCurved,
    planCenterline,
    planEdges,
    rows,
    graphPoints,
  };
}

function exportRowsAsWorkbook(rows, name) {
  const workbook = XLSX.utils.book_new();
  const sheet = XLSX.utils.aoa_to_sheet(rows);
  XLSX.utils.book_append_sheet(workbook, sheet, "Results");
  const output = XLSX.write(workbook, { bookType: "xlsx", type: "array" });
  const url = URL.createObjectURL(new Blob([output], { type: "application/octet-stream" }));
  triggerDownload(url, name);
}

function sortedSpans() {
  return Object.keys(state.spanToGirders).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
}

function sortedGirders(spanValue) {
  return Array.from(state.spanToGirders[spanValue] ?? []).sort((a, b) =>
    a.localeCompare(b, undefined, { numeric: true }),
  );
}

function populateGirderSelect(spanValue, selectEl) {
  if (!selectEl) return;

  const girders = sortedGirders(spanValue);
  if (!girders.length) {
    selectEl.innerHTML = '<option value="">(No girders found)</option>';
    selectEl.disabled = true;
    return;
  }

  const previous = selectEl.value;
  selectEl.disabled = false;
  selectEl.innerHTML = "";
  girders.forEach((girder) => {
    const option = document.createElement("option");
    option.value = girder;
    option.textContent = girder;
    selectEl.appendChild(option);
  });
  selectEl.value = girders.includes(previous) ? previous : girders[0];
}

function populateSpanGirderPair(spanSelect, girderSelect) {
  if (!spanSelect || !girderSelect) return;

  const spans = sortedSpans();
  if (!spans.length) {
    spanSelect.innerHTML = '<option value="">(Run calculation first)</option>';
    girderSelect.innerHTML = '<option value="">(Run calculation first)</option>';
    spanSelect.disabled = true;
    girderSelect.disabled = true;
    return;
  }

  const previous = spanSelect.value;
  spanSelect.disabled = false;
  spanSelect.innerHTML = "";
  spans.forEach((span) => {
    const option = document.createElement("option");
    option.value = span;
    option.textContent = span;
    spanSelect.appendChild(option);
  });

  spanSelect.value = spans.includes(previous) ? previous : spans[0];
  populateGirderSelect(spanSelect.value, girderSelect);
}

function populateGraphSelectors() {
  populateSpanGirderPair(ui.graphSpanSelect, ui.graphGirderSelect);
  populateSpanGirderPair(ui.deckSpanSelect, ui.deckGirderSelect);
}

function renderProfileChart() {
  const span = ui.graphSpanSelect.value;
  const girder = ui.graphGirderSelect.value;
  if (!span || !girder) return;

  const key = `${span}||${girder}`;
  const profile = state.profiles[key];
  if (!profile?.length) return;

  const x = profile.map((point) => point.station);
  const y = profile.map((point) => Math.abs(point.deflectionIn));

  Plotly.newPlot(
    ui.profileChart,
    [
      {
        x,
        y,
        mode: "lines+markers",
        hovertemplate: "Interval %{customdata[0]}<br>Station = %{x:.2f} ft<br>Deflection = %{customdata[1]:.3f} in<extra></extra>",
        customdata: profile.map((point) => [point.interval, point.deflectionIn]),
        line: { width: 3, color: "#0d6efd" },
        marker: { size: 8, color: "#0d6efd" },
      },
    ],
    {
      ...PLOTLY_LAYOUT,
      title: `<b>Span ${span} — Girder ${girder}</b>`,
      xaxis: { title: "Length along girder (ft)", zeroline: false },
      yaxis: { title: "Deflection (in)" },
      margin: { t: 60, r: 25, b: 60, l: 60 },
      paper_bgcolor: "#fcfdff",
      plot_bgcolor: "#fcfdff",
      showlegend: false,
    },
    PLOTLY_CONFIG,
  );
  enableZoomWindow(ui.profileChart);
}

// Target size (px) of a plan-view grid cell on screen.
const PLAN_GRID_PX = 80;

/** The 1, 2 or 5 x 10^n step nearest above `target`. */
function niceStep(target) {
  if (!(target > 0)) return 1;
  const power = 10 ** Math.floor(Math.log10(target));
  const scaled = target / power;
  return (scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10) * power;
}

/**
 * Plan views keep true proportions (1 ft north = 1 ft east on screen), so the
 * same grid step on both axes gives square cells. The step is picked from the
 * visible area -- about PLAN_GRID_PX per cell -- so the grid stays square and
 * readable at any zoom.
 */
function fitSquareGrid(gd) {
  const xa = gd._fullLayout?.xaxis;
  const ya = gd._fullLayout?.yaxis;
  if (!xa?._length || !ya?._length) return;
  const ftPerPx = Math.max(
    Math.abs(xa.range[1] - xa.range[0]) / xa._length,
    Math.abs(ya.range[1] - ya.range[0]) / ya._length,
  );
  const step = niceStep(ftPerPx * PLAN_GRID_PX);
  if (xa.dtick === step && ya.dtick === step) return;
  const decimals = step >= 1 ? 0 : Math.ceil(-Math.log10(step) - 1e-9);
  const format = `.${decimals}f`;
  Plotly.relayout(gd, {
    "xaxis.dtick": step,
    "yaxis.dtick": step,
    "xaxis.tick0": 0,
    "yaxis.tick0": 0,
    "xaxis.tickformat": format,
    "yaxis.tickformat": format,
  });
}

/** Keeps a plan view's grid square after every render, zoom, pan, or resize. */
function enableSquareGrid(gd) {
  fitSquareGrid(gd);
  const refit = (update) => {
    if (update && "xaxis.dtick" in update) return; // our own relayout
    fitSquareGrid(gd);
  };
  // The zoom window applies its box with Plotly.update, which reports
  // plotly_update rather than plotly_relayout.
  gd.on("plotly_relayout", refit);
  gd.removeAllListeners?.("plotly_update");
  gd.on("plotly_update", () => fitSquareGrid(gd));
}

function renderPlanChart() {
  const span = ui.graphSpanSelect.value;
  const selectedGirder = ui.graphGirderSelect.value;
  const spans = Object.keys(state.spanToGirders).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  if (!spans.length) return;

  const traces = spans
    .flatMap((spanValue) => {
      const girders = Array.from(state.spanToGirders[spanValue] ?? []).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
      return girders.map((girder) => {
        const key = `${spanValue}||${girder}`;
        const geo = state.girderGeometry[key];
        if (!geo) return null;
        const isSelected = spanValue === span && girder === selectedGirder;
        return {
          x: geo.planCenterline.map((point) => point.e),
          y: geo.planCenterline.map((point) => point.n),
          mode: "lines+markers",
          line: {
            width: isSelected ? 6 : 3,
            color: isSelected ? "#d63384" : "#6c757d",
          },
          marker: { size: isSelected ? 10 : 7 },
          name: `Span ${spanValue} — Girder ${girder}`,
          customdata: geo.planCenterline.map(() => [spanValue, girder]),
          hovertemplate: `Span ${spanValue}<br>Girder ${girder}<extra></extra>`,
        };
      });
    })
    .filter(Boolean);
  if (!traces.length) return;

  Plotly.newPlot(
    ui.planChart,
    traces,
    {
      ...PLOTLY_LAYOUT,
      title: "<b>Plan View for All Spans (N/E)</b>",
      xaxis: {
        title: { text: "Easting (ft)", standoff: 34 },
        tickformat: ".0f",
        exponentformat: "none",
        showexponent: "none",
        tickangle: -45,
        automargin: true,
      },
      yaxis: {
        title: { text: "Northing (ft)", standoff: 14 },
        scaleanchor: "x",
        scaleratio: 1,
        tickformat: ".0f",
        exponentformat: "none",
        showexponent: "none",
        automargin: true,
      },
      margin: { t: 60, r: 25, b: 115, l: 95 },
      paper_bgcolor: "#fcfdff",
      plot_bgcolor: "#fcfdff",
      showlegend: false,
    },
    PLOTLY_CONFIG,
  );
  enableZoomWindow(ui.planChart);
  enableSquareGrid(ui.planChart);
  bindChartClick(ui.planChart, onPlanChartClick);
}


function logLine(message) {
  state.logs.push(message);
  ui.logOutput.textContent = state.logs.join("\n");
}

function minMax(values) {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < values.length; i += 1) {
    const value = values[i];
    if (value < min) min = value;
    if (value > max) max = value;
  }
  return values.length ? { min, max } : null;
}

function girderPlanBounds() {
  const points = [];
  Object.values(state.girderGeometry).forEach((geometry) => {
    (geometry?.planCenterline ?? []).forEach((point) => points.push(point));
  });
  return points.length ? BridgeLandXml.boundsOf(points) : null;
}

function readTextFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (event) => resolve(event.target.result);
    reader.onerror = () => reject(new Error("Failed to read the selected file."));
    reader.readAsText(file);
  });
}

async function loadDtmSurface() {
  const [file] = ui.dtmFileInput.files;
  state.deflectedDeck = null;

  if (!file) {
    state.dtm = null;
    state.dtmTin = null;
    state.dtmBoundary = null;
    state.planRevision += 1;
    ui.dtmUploadStatus.textContent = "";
    refreshSections();
    renderDeflectedDeckChart();
    return;
  }

  const text = await readTextFile(file);
  const { surfaces } = BridgeLandXml.parseLandXml(text);
  const surface = surfaces[0];
  state.dtm = surface;
  // Reused verbatim in the exported surface so it keeps the source's units.
  state.dtmUnitsXml = (text.match(/<Units>[\s\S]*?<\/Units>/) || [null])[0];
  state.dtmFileName = file.name;
  state.dtmTin = BridgeIsopach.buildTinInterpolator(surface.points, surface.faces);
  state.dtmBoundary = null;
  state.planRevision += 1;

  ui.dtmUploadStatus.textContent = `Loaded "${surface.name}" - ${surface.points.length} points, ${surface.faces.length} faces.`;
  logLine(
    `DTM: loaded surface "${surface.name}" with ${surface.points.length} points and ${surface.faces.length} faces.`,
  );
  if (surfaces.length > 1) {
    logLine(`DTM: file contains ${surfaces.length} surfaces; using the first ("${surface.name}").`);
  }

  const bounds = girderPlanBounds();
  if (bounds && BridgeLandXml.detectSwappedAxes(surface.bounds, bounds)) {
    logLine(
      "DTM WARNING: surface coordinates only overlap the girders when N and E are swapped. " +
        "Check the exporter's axis order before trusting the result.",
    );
  }

  refreshSections();
  renderDeflectedDeckChart();
}

/** Blank means "use the girder spacing"; returns null in that case. */
function readOverhangOffset() {
  const raw = String(ui.overhangInput?.value ?? "").trim();
  if (!raw) return null;
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0) {
    throw new Error("Overhang offset must be a number of feet, 0 or greater (or blank to use the girder spacing).");
  }
  return value > 0 ? value : null;
}

/**
 * How the deck runs past the edge of deck: "slope" carries its cross slope out
 * (the default), "level" holds the edge-of-deck elevation. Either way the edge
 * of deck is a break line.
 */
function readOverhangSlope() {
  return ui.overhangSlopeSelect?.value === "level" ? "level" : "slope";
}

/** How the overhang grade reads in the log, notes and export descriptions. */
function describeOverhangSlope(overhangSlope) {
  return overhangSlope === "level" ? "held level from the edge of deck" : "carried at its cross slope";
}

/** Short chart label for the computed deck's extension past the DTM. */
function extensionLabel() {
  return state.deflectedDeck?.overhangSlope === "level" ? "held level" : "carried at cross slope";
}

// How much of the deck, inward from the DTM edge, sets the cross slope that
// is carried out to the overhang (screed) line.
const CROSS_SLOPE_RUN = 2;
// Tolerance (ft) when deciding whether a point is within the overhang.
const OVERHANG_TOLERANCE = 1e-3;

/**
 * Deck side edges -- the DTM boundary running along the given exterior
 * girders -- with outward normals. The overhang is measured perpendicular to
 * these edges, which may curve (they usually follow the alignment) even where
 * the girders are straight.
 */
function deckSidesFor(fascias) {
  if (!state.dtmBoundary) {
    state.dtmBoundary = BridgeSurfaceExport.tinBoundary(state.dtm.points, state.dtm.faces);
  }
  return BridgeSurfaceExport.deckSideEdges(state.dtmBoundary, fascias);
}

/**
 * Deck elevation at a deck-edge point and the cross slope carried out from it
 * (over the last CROSS_SLOPE_RUN ft of the model, perpendicular to the edge),
 * or a zero slope when the overhang is held level. `z(d)` gives the deck
 * elevation `d` ft beyond the edge.
 */
function deckEdgeProfile(tin, edgePoint, overhangSlope = "slope") {
  const nudge = 1e-4; // read just inside the edge, where the TIN is certain to answer
  // At a deck corner, straight inward runs along the end edge, where the TIN
  // has no answer; step a hair along the edge (either way) to stay on the deck.
  const tangent = { e: -edgePoint.normal.n, n: edgePoint.normal.e };
  const inward = (s) => {
    for (const t of [0, 0.01, -0.01]) {
      const z = tin.sample(
        edgePoint.e - edgePoint.normal.e * s + tangent.e * t,
        edgePoint.n - edgePoint.normal.n * s + tangent.n * t,
      );
      if (z !== null) return z;
    }
    return null;
  };
  const edgeZ = inward(nudge);
  if (edgeZ === null) return null;
  if (overhangSlope === "level") return { slope: 0, z: () => edgeZ };
  const backZ = inward(CROSS_SLOPE_RUN);
  const slope = backZ === null ? 0 : (edgeZ - backZ) / (CROSS_SLOPE_RUN - nudge);
  return { slope, z: (d) => edgeZ + slope * (d + nudge) };
}

/**
 * Deck elevations out to the overhang (screed) line. Inside the DTM this is
 * the TIN elevation. Between the deck edge and the screed line -- the
 * overhang offset beyond the edge, perpendicular to it -- the last
 * CROSS_SLOPE_RUN ft of deck is carried out at its own cross slope (or, with
 * `overhangSlope` "level", the edge elevation is held), so a surveyor gets
 * prorated elevations where the screed sits beyond the model. The edge of
 * deck is a break line: the overhang grade starts there and never blends
 * back into the model.
 */
function buildDeckSurface(tin, mesh, overhangOffset, overhangSlope = "slope") {
  const sides = overhangOffset !== null && mesh && tin ? deckSidesFor(mesh.overhangEdges) : null;
  const active = Boolean(sides?.edges.length);

  /** Deck-edge point and distance beyond it, if the point is in the overhang. */
  function locate(e, n) {
    const edge = BridgeSurfaceExport.closestOnDeckEdge(sides, e, n);
    if (!edge) return null;
    const dE = e - edge.e;
    const dN = n - edge.n;
    const along = dE * edge.normal.e + dN * edge.normal.n;
    const lateral = Math.abs(dE * edge.normal.n - dN * edge.normal.e);
    if (along <= 0 || along > overhangOffset + OVERHANG_TOLERANCE) return null;
    // Past the end of a deck side (e.g. beyond the abutment corner) the
    // closest edge point is the corner, off to one side: not overhang.
    if (lateral > OVERHANG_TOLERANCE + 0.05 * along) return null;
    return { edge, along };
  }

  /** The overhang carries the deflection it has at the deck edge (the fascia girder's). */
  const edgeMeshPoint = (edge) => ({ e: edge.e - edge.normal.e * 1e-4, n: edge.n - edge.normal.n * 1e-4 });

  return {
    extends: active,
    overhangSlope,
    sides,
    /** True where the deck exists only because of the overhang extension. */
    inOverhangBand(e, n) {
      return active && (!tin || tin.sample(e, n) === null) && locate(e, n) !== null;
    },
    /** Deflection to apply at a plan position, including across the overhang. */
    isopachAt(e, n) {
      const hit = mesh?.sample(e, n);
      if (hit) return hit.value;
      if (!active) return 0;
      const band = locate(e, n);
      if (!band) return 0;
      const point = edgeMeshPoint(band.edge);
      return mesh.sample(point.e, point.n)?.value ?? 0;
    },
    /** {z, extended, crossSlope, meshPoint?} or null where there is no deck. */
    sample(e, n) {
      const z = tin ? tin.sample(e, n) : null;
      if (z !== null) return { z, extended: false, crossSlope: null };
      if (!active) return null;
      const hit = locate(e, n);
      if (!hit) return null;
      const profile = deckEdgeProfile(tin, hit.edge, overhangSlope);
      if (!profile) return null;
      return {
        z: profile.z(hit.along),
        extended: true,
        crossSlope: profile.slope,
        meshPoint: edgeMeshPoint(hit.edge),
      };
    },
  };
}

function getDeckOutlineRings() {
  const surface = state.dtm;
  if (!surface) return [];

  const outer = surface.boundaries.filter((boundary) => boundary.type !== "island" && boundary.type !== "hole");
  if (outer.length) return outer.map((boundary) => boundary.points);

  if (surface.faces.length) {
    const rings = BridgeLandXml.extractBoundaryRings(surface.points, surface.faces);
    if (rings.length) return rings;
  }

  const hull = BridgeIsopach.convexHull(surface.points);
  return hull.length >= 3 ? [hull] : [];
}

/**
 * Declared hole/island boundaries, which getDeckOutlineRings leaves out when
 * the surface also declares an outer boundary. (Rings traced from the TIN
 * already include its holes.)
 */
function getDeckExclusionRings() {
  const boundaries = state.dtm?.boundaries ?? [];
  const isExclusion = (boundary) => boundary.type === "island" || boundary.type === "hole";
  if (!boundaries.some((boundary) => !isExclusion(boundary))) return [];
  return boundaries.filter(isExclusion).map((boundary) => boundary.points);
}

function logDeckReferenceStats() {
  const deckZ = minMax(state.dtm.points.map((point) => point.z));
  const isopach = minMax(state.deflectedDeck.points.map((point) => point.isopach));

  const undeflectedGirderZ = [];
  for (let i = 1; i < state.topOfGirderPoints.length; i += 1) {
    const row = state.topOfGirderPoints[i];
    undeflectedGirderZ.push(row[2] - row[4]);
  }
  const girderZ = minMax(undeflectedGirderZ);

  if (deckZ) logLine(`QC: DTM deck elevation range ${deckZ.min.toFixed(3)} to ${deckZ.max.toFixed(3)} ft.`);
  if (girderZ) {
    logLine(
      `QC: undeflected top-of-girder range ${girderZ.min.toFixed(3)} to ${girderZ.max.toFixed(3)} ft ` +
        "(deck should sit above this by the haunch + slab thickness).",
    );
  }
  if (isopach) logLine(`QC: deflection applied ranges ${isopach.min.toFixed(3)} to ${isopach.max.toFixed(3)} ft.`);

  if (deckZ && girderZ && deckZ.max < girderZ.min) {
    logLine(
      "QC WARNING: the entire DTM sits below the top of girder. The uploaded surface is probably not the " +
        "theoretical top of deck (or the units/datum differ), so the deflected elevations will be wrong.",
    );
  }
}

function computeDeflectedDeck() {
  if (!state.topOfGirderPoints.length) {
    window.alert("Please calculate the top-of-girder points first (Girder Calcs tab).");
    return false;
  }
  if (!state.dtm) {
    window.alert("Please upload the top-of-deck DTM XML surface first.");
    return false;
  }

  let overhangOffset;
  try {
    overhangOffset = readOverhangOffset();
  } catch (error) {
    window.alert(error.message);
    return false;
  }
  const overhangSlope = readOverhangSlope();

  // The overhang offset is measured perpendicular to the edge of deck (where
  // the DTM ends), which may curve while the girders are straight. For each
  // exterior girder interval, take the nearest point on its deck edge and go
  // out the offset along that edge's outward normal.
  const edgeRanges = [];
  const screedGeometry = new Map();
  const overhangPoints =
    overhangOffset === null
      ? undefined
      : ({ span, girder, points, normals }) => {
          const sides = deckSidesFor([{ fascia: points, outward: normals }]);
          if (!sides.edges.length) {
            logLine(
              `Overhang WARNING: Span ${span}, Girder ${girder}: no deck edge was found alongside this exterior ` +
                "girder, so the overhang offset is measured from the girder centerline instead.",
            );
            return points.map((point, index) => ({
              e: point.e + normals[index].e * overhangOffset,
              n: point.n + normals[index].n * overhangOffset,
            }));
          }
          const edges = points.map((point) => BridgeSurfaceExport.closestOnDeckEdge(sides, point.e, point.n));
          screedGeometry.set(`${span}||${girder}`, edges);
          const toEdge = edges.map((edge) => edge.distance);
          edgeRanges.push({ span, girder, min: Math.min(...toEdge), max: Math.max(...toEdge) });
          return edges.map((edge) => ({
            e: edge.e + edge.normal.e * overhangOffset,
            n: edge.n + edge.normal.n * overhangOffset,
          }));
        };

  const mesh = BridgeIsopach.buildIsopachMesh({
    spanToGirders: state.spanToGirders,
    girderGeometry: state.girderGeometry,
    profiles: state.profiles,
    overhangOffset,
    overhangPoints,
  });
  mesh.warnings.forEach((warning) => logLine(`Isopach WARNING: ${warning}`));
  if (overhangOffset === null) {
    logLine("Isopach: no overhang offset given; the deck is not extended past the DTM.");
  } else {
    logLine(
      `Isopach: overhang edge set ${overhangOffset.toFixed(3)} ft beyond the edge of deck (DTM edge), ` +
        `measured perpendicular to the deck edge; the deck is ${describeOverhangSlope(overhangSlope)} ` +
        "(the edge of deck is a break line).",
    );
    edgeRanges.forEach((range) => {
      logLine(
        `Overhang: Span ${range.span}, Girder ${range.girder}: edge of deck ${range.min.toFixed(3)}` +
          (range.max - range.min > 0.001 ? ` to ${range.max.toFixed(3)}` : "") +
          " ft from the girder centerline.",
      );
    });
  }

  if (!mesh.triangleCount) {
    window.alert("No deflection surface could be built. At least one span needs two or more girders.");
    return false;
  }

  const girderBounds = girderPlanBounds();
  const deckBounds = state.dtm.bounds;
  const overlaps =
    girderBounds &&
    deckBounds &&
    girderBounds.minE <= deckBounds.maxE &&
    deckBounds.minE <= girderBounds.maxE &&
    girderBounds.minN <= deckBounds.maxN &&
    deckBounds.minN <= girderBounds.maxN;

  if (!overlaps) {
    logLine("Deflected deck ERROR: DTM extents do not overlap the girder extents; nothing was computed.");
    window.alert(
      "The DTM surface does not overlap the girder footprint. Confirm both files use the same coordinate system.",
    );
    return false;
  }

  const points = state.dtm.points.map((point) => {
    const hit = mesh.sample(point.e, point.n);
    const isopach = hit ? hit.value : 0;
    return {
      id: point.id,
      name: point.name,
      n: point.n,
      e: point.e,
      originalZ: point.z,
      isopach,
      deflectedZ: point.z + isopach,
      spanKey: hit ? hit.spanKey : null,
      girderKey: hit ? hit.girderKey : null,
    };
  });

  const inside = points.reduce((total, point) => total + (point.spanKey === null ? 0 : 1), 0);

  // A deck DTM is often built from a few longitudinal feature lines (edges and
  // PGL), so it may have no vertex anywhere near an interior girder. Sample the
  // deflected surface directly above the girder's left and right edges instead
  // -- the same plan positions and names as the Top of Girder export -- so each
  // girder has elevations to report regardless of where the DTM places vertices.
  const tin = state.dtmTin;
  const girderPoints = {};
  let sampledGirders = 0;

  sortedSpans().forEach((span) => {
    sortedGirders(span).forEach((girder) => {
      const key = `${span}||${girder}`;
      const edges = state.girderGeometry[key]?.planEdges;
      if (!edges?.length) return;

      const sampled = [];
      edges.forEach((point, index) => {
        const interval = Math.floor(index / 2);
        const deckZ = tin.sample(point.e, point.n);
        if (deckZ === null) return;
        const hit = mesh.sample(point.e, point.n);
        const isopach = hit ? hit.value : 0;
        sampled.push({
          e: point.e,
          n: point.n,
          interval,
          side: point.side,
          code: `${formatSpan(span)}${formatGirder(girder)}${formatInterval(interval)}${point.side}`,
          originalZ: deckZ,
          isopach,
          deflectedZ: deckZ + isopach,
        });
      });

      if (sampled.length) {
        girderPoints[key] = sampled;
        sampledGirders += 1;
      }
    });
  });

  // Screed points: the overhang edge at every girder interval, with the deck
  // carried out at its cross slope (or held level) and the fascia girder's
  // deflection added.
  const deckSurface = buildDeckSurface(tin, mesh, overhangOffset, overhangSlope);
  const edgePoints = [];
  let extendedEdgePoints = 0;
  if (deckSurface.extends) {
    mesh.overhangEdges.forEach((edge) => {
      const key = `${edge.span}||${edge.girder}`;
      const profile = state.profiles[key] ?? [];
      const deckEdges = screedGeometry.get(key);
      edge.points.forEach((point, interval) => {
        if (!profile[interval]) return;
        // Carry the deck out from the deck-edge point this screed point was
        // placed from; without one (no deck edge found) fall back to sampling.
        const deckEdge = deckEdges?.[interval];
        const edgeProfile = deckEdge ? deckEdgeProfile(tin, deckEdge, overhangSlope) : null;
        const deck = edgeProfile
          ? { z: edgeProfile.z(overhangOffset), extended: true, crossSlope: edgeProfile.slope }
          : deckSurface.sample(point.e, point.n);
        if (!deck) return;
        // The overhang row carries the fascia value by construction; read it
        // from the profile rather than sampling right on the mesh boundary.
        const isopach = profile[interval].deflectionIn / 12;
        if (deck.extended) extendedEdgePoints += 1;
        edgePoints.push({
          span: edge.span,
          girder: edge.girder,
          interval,
          e: point.e,
          n: point.n,
          originalZ: deck.z,
          extended: deck.extended,
          crossSlope: deck.crossSlope,
          isopach,
          deflectedZ: deck.z + isopach,
        });
      });
    });
  }

  state.isopachMesh = mesh;
  state.deflectedDeck = { points, inside, girderPoints, overhangOffset, overhangSlope, deckSurface, edgePoints };
  state.planRevision += 1;

  if (deckSurface.extends) {
    const expected = mesh.overhangEdges.reduce((total, edge) => total + edge.points.length, 0);
    logLine(
      `Overhang: ${edgePoints.length} of ${expected} screed points sampled, ${extendedEdgePoints} of them past the ` +
        "DTM edge (deck " +
        (overhangSlope === "level"
          ? "held level at the edge-of-deck elevation)."
          : `carried out at its cross slope over the last ${CROSS_SLOPE_RUN} ft of the model).`),
    );
    if (edgePoints.length < expected) {
      logLine(
        "Overhang WARNING: some screed points have no deck elevation; the DTM does not reach the exterior girder there.",
      );
    }
  }

  logLine(
    `Deflected deck: ${points.length} DTM points - ${inside} inside the deflected girder area, ` +
      `${points.length - inside} outside (no deflection applied, original elevation kept).`,
  );
  if (tin.triangleCount) {
    logLine(
      `Deflected deck: sampled deck elevations above the left and right edges of ${sampledGirders} girders ` +
        "(same positions and names as the Top of Girder points).",
    );
  } else {
    logLine(
      "Deflected deck NOTE: the DTM has no TIN faces, so deck elevations could not be sampled along the " +
        "girder centerlines. Only the surface's own points are shown.",
    );
  }
  logDeckReferenceStats();

  refreshSections();
  renderDeflectedDeckChart();
  return true;
}

// Pale = little or no deflection, deep red = the most (Plotly's YlOrRd, reversed).
const DEFLECTION_COLORSCALE = [
  [0, "#ffffcc"],
  [0.125, "#ffeda0"],
  [0.25, "#fed976"],
  [0.375, "#feb24c"],
  [0.5, "#fd8d3c"],
  [0.625, "#fc4e2a"],
  [0.75, "#e31a1c"],
  [0.875, "#bd0026"],
  [1, "#800026"],
];

// Low = dark blue, high = yellow (Plotly's Viridis).
const ELEVATION_COLORSCALE = [
  [0, "#440154"],
  [0.125, "#472d7b"],
  [0.25, "#3b528b"],
  [0.375, "#2c728e"],
  [0.5, "#21918c"],
  [0.625, "#28ae80"],
  [0.75, "#5ec962"],
  [0.875, "#addc30"],
  [1, "#fde725"],
];

// Surfaces the deck plan view can shade. The isopach and deflected deck need
// Compute deck; the DTM can be shown as soon as it is loaded.
const PLAN_SURFACES = {
  isopach: { title: "Deflection (ft)", hover: "Deflection", colorscale: DEFLECTION_COLORSCALE },
  deflected: { title: "Deflected deck elev. (ft)", hover: "Deflected", colorscale: ELEVATION_COLORSCALE },
  dtm: { title: "DTM elev. (ft)", hover: "DTM", colorscale: ELEVATION_COLORSCALE },
};

// Grid cells along the longer side of the deck, and image pixels along it.
const PLAN_SURFACE_CELLS = 240;
const PLAN_SURFACE_PIXELS = 3000;

function selectedPlanSurface() {
  const mode = ui.planSurfaceSelect?.value;
  return PLAN_SURFACES[mode] ? mode : "isopach";
}

/** Plotly-style colorscale lookup: t in [0, 1] -> [r, g, b]. */
function colorscaleRgb(colorscale, t) {
  const clamped = Math.max(0, Math.min(1, t));
  const hex = (color) => [1, 3, 5].map((i) => parseInt(color.slice(i, i + 2), 16));
  for (let i = 1; i < colorscale.length; i += 1) {
    const [t1, c1] = colorscale[i];
    if (clamped > t1) continue;
    const [t0, c0] = colorscale[i - 1];
    const f = t1 > t0 ? (clamped - t0) / (t1 - t0) : 0;
    const a = hex(c0);
    const b = hex(c1);
    return a.map((value, k) => Math.round(value + (b[k] - value) * f));
  }
  return hex(colorscale[colorscale.length - 1][1]);
}

/**
 * Plan value function and domain for a surface mode, or null when that
 * surface is not available yet. Areas past the span ends read 0 deflection,
 * which is physically correct -- the deflection parabola is zero at every
 * support.
 */
function planSurfaceSampler(mode, rings) {
  if (mode === "dtm") {
    const tin = state.dtmTin;
    if (!tin?.triangleCount) return null;
    return {
      inside: (e, n) => BridgeIsopach.pointInRings(e, n, rings),
      value: (e, n) => tin.sample(e, n),
      extendsDeck: false,
    };
  }

  const mesh = state.isopachMesh;
  const surface = state.deflectedDeck?.deckSurface;
  if (!mesh || !surface) return null;
  const extendsDeck = Boolean(surface.extends);
  const isopachAt = extendsDeck ? (e, n) => surface.isopachAt(e, n) : (e, n) => mesh.sample(e, n)?.value ?? 0;
  const inside = (e, n) => BridgeIsopach.pointInRings(e, n, rings) || (extendsDeck && surface.inOverhangBand(e, n));

  if (mode === "deflected") {
    return {
      inside,
      value: (e, n) => {
        const deck = surface.sample(e, n);
        return deck ? deck.z + isopachAt(e, n) : null;
      },
      extendsDeck,
    };
  }
  return { inside, value: isopachAt, extendsDeck };
}

/**
 * Fills empty cells from their valid neighbours, a few cells deep, so the
 * smoothed image has real colour right up to (and just past) the deck edge
 * before it is clipped there.
 */
function dilateGrid(z, passes) {
  let grid = z;
  for (let pass = 0; pass < passes; pass += 1) {
    grid = grid.map((row, j) =>
      row.map((value, i) => {
        if (value !== null) return value;
        let sum = 0;
        let count = 0;
        for (let dj = -1; dj <= 1; dj += 1) {
          for (let di = -1; di <= 1; di += 1) {
            const neighbour = grid[j + dj]?.[i + di];
            if (neighbour === null || neighbour === undefined) continue;
            sum += neighbour;
            count += 1;
          }
        }
        return count ? sum / count : null;
      }),
    );
  }
  return grid;
}

/**
 * The deck's plan footprint as canvas fills: the DTM outline rings (even-odd,
 * so holes stay empty) and, with an overhang, one quad per deck side edge out
 * to the screed line.
 */
function planSurfaceMaskShapes(rings, extendsDeck) {
  const shapes = [{ rings, evenOdd: true }];
  const deck = state.deflectedDeck;
  if (!extendsDeck || !deck?.deckSurface?.sides) return shapes;

  // A hair past the offset so the clip does not shave the screed line.
  const offset = deck.overhangOffset + OVERHANG_TOLERANCE;
  const quads = [];
  deck.deckSurface.sides.edges.forEach((edge) => {
    if (!edge.na || !edge.nb) return;
    const quad = [
      edge.a,
      edge.b,
      { e: edge.b.e + edge.nb.e * offset, n: edge.b.n + edge.nb.n * offset },
      { e: edge.a.e + edge.na.e * offset, n: edge.a.n + edge.na.n * offset },
    ];
    // Wind every quad the same way so one nonzero fill unions them. Filling
    // them one by one leaves anti-aliased seams along their shared sides.
    let area = 0;
    quad.forEach((p, i) => {
      const q = quad[(i + 1) % quad.length];
      area += p.e * q.n - q.e * p.n;
    });
    quads.push(area < 0 ? quad.reverse() : quad);
  });
  if (quads.length) shapes.push({ rings: quads, evenOdd: false });
  return shapes;
}

/**
 * Paints the grid as a smooth image and cuts it to the exact deck footprint.
 * A heatmap trace can only show whole grid cells, which leaves a staircase
 * along any deck edge that is not parallel to an axis.
 */
function renderPlanSurfaceImage(grid, colorscale, zmin, zmax, shapes) {
  const { xs, ys, step, filled } = grid;
  const cols = xs.length;
  const rows = ys.length;
  const range = zmax - zmin;

  const cells = document.createElement("canvas");
  cells.width = cols;
  cells.height = rows;
  const cellContext = cells.getContext("2d");
  const pixels = cellContext.createImageData(cols, rows);
  for (let j = 0; j < rows; j += 1) {
    // Canvas rows run top-down; northings run bottom-up.
    const rowOffset = (rows - 1 - j) * cols;
    for (let i = 0; i < cols; i += 1) {
      const value = filled[j][i];
      if (value === null) continue;
      const [r, g, b] = colorscaleRgb(colorscale, range > 0 ? (value - zmin) / range : 0.5);
      const index = (rowOffset + i) * 4;
      pixels.data[index] = r;
      pixels.data[index + 1] = g;
      pixels.data[index + 2] = b;
      pixels.data[index + 3] = 255;
    }
  }
  cellContext.putImageData(pixels, 0, 0);

  // Each cell's centre sits on its grid point, so the image spans half a
  // cell past the first and last grid points.
  const x0 = xs[0] - step / 2;
  const yTop = ys[rows - 1] + step / 2;
  const sizeX = cols * step;
  const sizeY = rows * step;
  const scale = PLAN_SURFACE_PIXELS / Math.max(cols, rows);
  const width = Math.max(1, Math.round(cols * scale));
  const height = Math.max(1, Math.round(rows * scale));

  const image = document.createElement("canvas");
  image.width = width;
  image.height = height;
  const context = image.getContext("2d");
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(cells, 0, 0, width, height);

  const mask = document.createElement("canvas");
  mask.width = width;
  mask.height = height;
  const maskContext = mask.getContext("2d");
  maskContext.fillStyle = "#000";
  const toPixel = (point) => [((point.e - x0) / sizeX) * width, ((yTop - point.n) / sizeY) * height];
  shapes.forEach((shape) => {
    maskContext.beginPath();
    shape.rings.forEach((ring) => {
      ring.forEach((point, index) => {
        const [px, py] = toPixel(point);
        if (index === 0) maskContext.moveTo(px, py);
        else maskContext.lineTo(px, py);
      });
      maskContext.closePath();
    });
    maskContext.fill(shape.evenOdd ? "evenodd" : "nonzero");
  });

  context.globalCompositeOperation = "destination-in";
  context.drawImage(mask, 0, 0);

  return {
    source: image.toDataURL("image/png"),
    xref: "x",
    yref: "y",
    x: x0,
    y: yTop,
    sizex: sizeX,
    sizey: sizeY,
    sizing: "stretch",
    xanchor: "left",
    yanchor: "top",
    layer: "below",
  };
}

// Painting the image is the slow part; reuse it until the data or mode changes.
let planSurfaceCache = { key: null, value: null };

/**
 * Samples the selected surface on a regular grid over the deck, so it is
 * shown across the whole deck rather than only where the DTM happens to
 * place vertices. Returns the hover grid (null off the deck), its value
 * range, and the clipped image to draw.
 */
function buildPlanSurface(rings, mode) {
  const key = `${state.planRevision}|${mode}`;
  if (planSurfaceCache.key === key) return planSurfaceCache.value;

  const value = computePlanSurface(rings, mode);
  planSurfaceCache = { key, value };
  return value;
}

function computePlanSurface(outlineRings, mode) {
  if (!outlineRings.length) return null;
  // Even-odd over the outline and any declared holes, so holes stay empty.
  const rings = outlineRings.concat(getDeckExclusionRings());
  const sampler = planSurfaceSampler(mode, rings);
  if (!sampler) return null;
  const shapes = planSurfaceMaskShapes(rings, sampler.extendsDeck);
  const grid = samplePlanGrid(shapes, sampler);
  if (!grid) return null;

  let zmin = Infinity;
  let zmax = -Infinity;
  grid.z.forEach((row) =>
    row.forEach((value) => {
      if (value === null) return;
      if (value < zmin) zmin = value;
      if (value > zmax) zmax = value;
    }),
  );
  if (!Number.isFinite(zmin)) return null;

  const surface = PLAN_SURFACES[mode];
  grid.filled = dilateGrid(grid.z, 3);
  return {
    ...grid,
    zmin,
    zmax,
    inside: sampler.inside,
    contours: new Map(), // by interval
    image: renderPlanSurfaceImage(grid, surface.colorscale, zmin, zmax, shapes),
  };
}

// Heavier, labelled index contour every this many intervals.
const CONTOUR_INDEX_EVERY = 5;
// More levels than this are unreadable at any zoom (and slow to trace).
const MAX_CONTOUR_LEVELS = 300;

/** Blank or invalid gives null; the caller reports it. */
function readContourInterval() {
  const value = Number(ui.contourIntervalInput?.value);
  return Number.isFinite(value) && value > 0 ? value : null;
}

/**
 * Marching squares over the filled grid: every contour at `level`, as
 * polylines of {e, n}. Crossings are keyed by the grid edge they sit on, so
 * the segments of neighbouring cells chain into continuous lines.
 */
function traceContourLevel(surface, level) {
  const { xs, ys, filled } = surface;
  const segments = [];
  const points = new Map();

  const crossing = (key, e0, n0, v0, e1, n1, v1) => {
    if (!points.has(key)) {
      const t = (level - v0) / (v1 - v0);
      points.set(key, { e: e0 + (e1 - e0) * t, n: n0 + (n1 - n0) * t });
    }
    return key;
  };

  for (let j = 0; j + 1 < ys.length; j += 1) {
    for (let i = 0; i + 1 < xs.length; i += 1) {
      const v00 = filled[j][i];
      const v10 = filled[j][i + 1];
      const v11 = filled[j + 1][i + 1];
      const v01 = filled[j + 1][i];
      if (v00 === null || v10 === null || v11 === null || v01 === null) continue;

      const code = (v00 >= level ? 1 : 0) | (v10 >= level ? 2 : 0) | (v11 >= level ? 4 : 0) | (v01 >= level ? 8 : 0);
      if (code === 0 || code === 15) continue;

      const [e0, e1, n0, n1] = [xs[i], xs[i + 1], ys[j], ys[j + 1]];
      const bottom = () => crossing(`h${i},${j}`, e0, n0, v00, e1, n0, v10);
      const right = () => crossing(`v${i + 1},${j}`, e1, n0, v10, e1, n1, v11);
      const top = () => crossing(`h${i},${j + 1}`, e0, n1, v01, e1, n1, v11);
      const left = () => crossing(`v${i},${j}`, e0, n0, v00, e0, n1, v01);
      const add = (a, b) => segments.push([a(), b()]);

      switch (code) {
        case 1: case 14: add(left, bottom); break;
        case 2: case 13: add(bottom, right); break;
        case 3: case 12: add(left, right); break;
        case 4: case 11: add(right, top); break;
        case 6: case 9: add(bottom, top); break;
        case 7: case 8: add(left, top); break;
        case 5: case 10: {
          // Saddle: the cell centre decides which corners connect.
          const centreAbove = (v00 + v10 + v11 + v01) / 4 >= level;
          if ((code === 5) === centreAbove) {
            add(left, top);
            add(bottom, right);
          } else {
            add(left, bottom);
            add(right, top);
          }
          break;
        }
        default: break;
      }
    }
  }

  // Chain the segments through their shared edge keys.
  const byKey = new Map();
  segments.forEach((segment, index) => {
    segment.forEach((key) => {
      if (!byKey.has(key)) byKey.set(key, []);
      byKey.get(key).push(index);
    });
  });
  const used = new Array(segments.length).fill(false);
  const walk = (key, from) => {
    const keys = [];
    let current = key;
    let previous = from;
    for (;;) {
      const next = (byKey.get(current) || []).find((index) => !used[index] && index !== previous);
      if (next === undefined) break;
      used[next] = true;
      const [a, b] = segments[next];
      current = a === current ? b : a;
      previous = next;
      keys.push(current);
    }
    return keys;
  };

  const lines = [];
  segments.forEach((segment, index) => {
    if (used[index]) return;
    used[index] = true;
    const forward = walk(segment[1], index);
    const backward = walk(segment[0], index);
    const keys = backward.reverse().concat(segment, forward);
    lines.push(keys.map((key) => points.get(key)));
  });
  return lines;
}

/**
 * Cuts contour lines at the deck edge. The filled grid runs a few cells past
 * the deck so lines reach it; where a line leaves the deck, the crossing is
 * found by bisection so the line stops right on the edge.
 */
function clipToDeck(lines, inside) {
  const pieces = [];
  const edgePoint = (a, b) => {
    // a is on the deck, b is off it.
    let lo = 0;
    let hi = 1;
    for (let k = 0; k < 12; k += 1) {
      const mid = (lo + hi) / 2;
      if (inside(a.e + (b.e - a.e) * mid, a.n + (b.n - a.n) * mid)) lo = mid;
      else hi = mid;
    }
    return { e: a.e + (b.e - a.e) * lo, n: a.n + (b.n - a.n) * lo };
  };

  lines.forEach((line) => {
    let piece = [];
    let previous = null;
    let previousInside = false;
    line.forEach((point) => {
      const isInside = inside(point.e, point.n);
      if (isInside && !previousInside && previous) piece.push(edgePoint(point, previous));
      if (isInside) piece.push(point);
      if (!isInside && previousInside) {
        piece.push(edgePoint(previous, point));
        if (piece.length > 1) pieces.push(piece);
        piece = [];
      }
      previous = point;
      previousInside = isInside;
    });
    if (piece.length > 1) pieces.push(piece);
  });
  return pieces;
}

/**
 * Contour overlay for the shaded surface: minor and index lines plus labels
 * on the index contours (or on every contour when no index one falls on the
 * deck). Traced once per surface and interval.
 */
function buildPlanContours(surface, interval) {
  const first = Math.ceil(surface.zmin / interval - 1e-9);
  const last = Math.floor(surface.zmax / interval + 1e-9);
  const levels = last - first + 1;
  if (levels > MAX_CONTOUR_LEVELS) return { tooMany: levels };
  if (surface.contours.has(interval)) return surface.contours.get(interval);

  const decimals = Math.min(4, (String(interval).split(".")[1] || "").length);
  const contours = [];
  for (let k = first; k <= last; k += 1) {
    const level = k * interval;
    const pieces = clipToDeck(traceContourLevel(surface, level), surface.inside);
    if (pieces.length) contours.push({ level, index: k % CONTOUR_INDEX_EVERY === 0, pieces });
  }

  const result = { levels: contours.length, traces: contourTraces(contours, decimals, surface.step) };
  surface.contours.set(interval, result);
  return result;
}

function contourTraces(contours, decimals, step) {
  const lineTrace = (index) => {
    const x = [];
    const y = [];
    contours
      .filter((contour) => contour.index === index)
      .forEach((contour) =>
        contour.pieces.forEach((piece) => {
          piece.forEach((point) => {
            x.push(point.e);
            y.push(point.n);
          });
          x.push(null);
          y.push(null);
        }),
      );
    return {
      x,
      y,
      mode: "lines",
      line: { width: index ? 1.5 : 0.7, color: "rgba(33,37,41,0.7)" },
      name: index ? "Index contours" : "Contours",
      hoverinfo: "skip",
    };
  };

  // One label per long enough line, at its middle.
  const labelled = contours.some((contour) => contour.index) ? contours.filter((c) => c.index) : contours;
  const labels = { x: [], y: [], text: [] };
  labelled.forEach((contour) =>
    contour.pieces.forEach((piece) => {
      const lengths = [0];
      for (let i = 1; i < piece.length; i += 1) {
        lengths.push(lengths[i - 1] + Math.hypot(piece[i].e - piece[i - 1].e, piece[i].n - piece[i - 1].n));
      }
      const total = lengths[lengths.length - 1];
      if (total < 15 * step) return;
      const i = lengths.findIndex((length) => length >= total / 2);
      labels.x.push(piece[i].e);
      labels.y.push(piece[i].n);
      labels.text.push(contour.level.toFixed(decimals));
    }),
  );

  return [
    lineTrace(false),
    lineTrace(true),
    {
      ...labels,
      mode: "text",
      textfont: { size: 10, color: "#212529" },
      name: "Contour labels",
      hoverinfo: "skip",
    },
  ];
}

/** Grid over the footprint `shapes` (which reach the screed line with an overhang). */
function samplePlanGrid(shapes, { inside, value }) {
  let minE = Infinity;
  let maxE = -Infinity;
  let minN = Infinity;
  let maxN = -Infinity;
  const include = (point) => {
    if (point.e < minE) minE = point.e;
    if (point.e > maxE) maxE = point.e;
    if (point.n < minN) minN = point.n;
    if (point.n > maxN) maxN = point.n;
  };
  shapes.forEach((shape) => shape.rings.forEach((ring) => ring.forEach(include)));

  const width = maxE - minE;
  const height = maxN - minN;
  if (!(width > 0) || !(height > 0)) return null;

  const step = Math.max(width, height) / PLAN_SURFACE_CELLS;
  // One extra cell all round so the image reaches past the deck edge.
  const cols = Math.max(2, Math.ceil(width / step) + 3);
  const rows = Math.max(2, Math.ceil(height / step) + 3);

  const xs = Array.from({ length: cols }, (_, i) => minE + (i - 1) * step);
  const ys = Array.from({ length: rows }, (_, j) => minN + (j - 1) * step);
  const z = ys.map((n) => xs.map((e) => (inside(e, n) ? value(e, n) : null)));

  return { xs, ys, z, step };
}

/** Adds the contour lines for the shaded surface, if turned on, and reports on them. */
function addContourOverlay(traces, shaded) {
  const status = ui.contourStatus;
  const setStatus = (text) => {
    if (status) status.textContent = text;
  };
  if (!ui.contoursToggle?.checked || !shaded) {
    setStatus("");
    return;
  }

  const interval = readContourInterval();
  if (interval === null) {
    setStatus("Enter a contour interval greater than 0.");
    return;
  }
  const contours = buildPlanContours(shaded, interval);
  if (contours.tooMany) {
    setStatus(
      `A ${interval} ft interval gives ${contours.tooMany} contours over this surface; ` +
        `use a larger interval (at most ${MAX_CONTOUR_LEVELS} contours).`,
    );
    return;
  }
  contours.traces.forEach((trace) => traces.push(trace));
  setStatus(
    contours.levels
      ? `${contours.levels} contour(s) at ${interval} ft; every ${CONTOUR_INDEX_EVERY}th is heavier.`
      : `No ${interval} ft contour falls within this surface's range.`,
  );
}

function renderDeflectedDeckChart() {
  if (!ui.deckChart) return;

  const selectedSpan = ui.deckSpanSelect?.value ?? "";
  const selectedGirder = ui.deckGirderSelect?.value ?? "";
  const traces = [];
  const images = [];
  const outlineRings = getDeckOutlineRings();

  const mode = selectedPlanSurface();
  const planSurface = PLAN_SURFACES[mode];
  const shaded = buildPlanSurface(outlineRings, mode);
  if (shaded) {
    images.push(shaded.image);
    // The image carries the colour; this invisible heatmap supplies the hover
    // readout and the colorbar.
    traces.push({
      type: "heatmap",
      x: shaded.xs,
      y: shaded.ys,
      z: shaded.z,
      zmin: shaded.zmin,
      zmax: shaded.zmax,
      colorscale: planSurface.colorscale,
      opacity: 0,
      zsmooth: false,
      hoverongaps: false,
      showscale: true,
      // title.side defaults to "top", which sits right in the corner where
      // Plotly's modebar (zoom/pan/reset) floats, hiding it behind the text.
      colorbar: { title: { text: planSurface.title, side: "right" }, thickness: 12, tickformat: ".3f" },
      name: planSurface.hover,
      hovertemplate: `N %{y:.3f}<br>E %{x:.3f}<br>${planSurface.hover} %{z:.3f} ft<extra></extra>`,
    });
  }
  addContourOverlay(traces, shaded);

  outlineRings.forEach((ring, index) => {
    if (ring.length < 3) return;
    const closed = ring.concat([ring[0]]);
    traces.push({
      x: closed.map((point) => point.e),
      y: closed.map((point) => point.n),
      mode: "lines",
      line: { width: 2, color: "#212529" },
      name: index === 0 ? "Deck outline" : `Deck outline ${index + 1}`,
      hoverinfo: "skip",
    });
  });

  sortedSpans().forEach((span) => {
    sortedGirders(span).forEach((girder) => {
      const geometry = state.girderGeometry[`${span}||${girder}`];
      if (!geometry?.planCenterline?.length) return;
      const isSelected = span === selectedSpan && girder === selectedGirder;
      traces.push({
        x: geometry.planCenterline.map((point) => point.e),
        y: geometry.planCenterline.map((point) => point.n),
        mode: "lines",
        line: { width: isSelected ? 6 : 3, color: isSelected ? "#d63384" : "#6c757d" },
        name: `Span ${span} - Girder ${girder}`,
        customdata: geometry.planCenterline.map(() => [span, girder]),
        hovertemplate: `Span ${span}<br>Girder ${girder}<extra></extra>`,
      });
    });
  });

  // Overhang (screed) lines only exist when the user gave an offset; the
  // blank-offset mesh edge is an internal detail, not a deck edge.
  const screedEdges = state.deflectedDeck?.deckSurface?.extends ? state.isopachMesh.overhangEdges : [];
  const screedPoints = state.deflectedDeck?.edgePoints ?? [];
  screedEdges.forEach((edge) => {
    const points = screedPoints.filter((point) => point.span === edge.span && point.girder === edge.girder);
    traces.push({
      x: points.map((point) => point.e),
      y: points.map((point) => point.n),
      mode: "lines+markers",
      line: { width: 2, color: "#fd7e14", dash: "dash" },
      marker: { size: 6, color: "#fd7e14" },
      name: `Span ${edge.span} - overhang edge at Girder ${edge.girder}`,
      customdata: points.map((point) => [
        `${formatSpan(point.span)}${formatGirder(point.girder)}${formatInterval(point.interval)}OH`,
        point.originalZ,
        point.isopach,
        point.deflectedZ,
        point.extended ? extensionLabel() : "from DTM",
      ]),
      hovertemplate:
        "<b>%{customdata[0]}</b> (overhang edge)<br>N %{y:.3f}<br>E %{x:.3f}<br>" +
        "Deck %{customdata[1]:.3f} ft (%{customdata[4]})<br>Deflection %{customdata[2]:.3f} ft<br>" +
        "<b>Deflected %{customdata[3]:.3f} ft</b><extra></extra>",
    });
  });

  alignmentPlanTraces().forEach((trace) => traces.push(trace));

  const deck = state.deflectedDeck;
  if (deck) {
    // Plotly's hovertemplate parser does not accept the "+" sign flag, and
    // silently falls back to full precision when it sees one.
    const hover =
      "N %{y:.3f}<br>E %{x:.3f}<br>DTM %{customdata[0]:.3f} ft<br>" +
      "Deflection %{customdata[1]:.3f} ft<br><b>Deflected %{customdata[2]:.3f} ft</b><extra></extra>";
    const toCustomdata = (point) => [point.originalZ, point.isopach, point.deflectedZ];

    traces.push({
      type: "scattergl",
      x: deck.points.map((point) => point.e),
      y: deck.points.map((point) => point.n),
      mode: "markers",
      marker: { size: 4, color: "rgba(33,37,41,0.55)" },
      name: "DTM surface points",
      customdata: deck.points.map(toCustomdata),
      hovertemplate: hover,
    });

    // Deck elevations sampled along the selected girder, labelled in place.
    const selectedPoints = deck.girderPoints?.[`${selectedSpan}||${selectedGirder}`] ?? [];
    if (selectedPoints.length) {
      traces.push({
        x: selectedPoints.map((point) => point.e),
        y: selectedPoints.map((point) => point.n),
        mode: "markers+text",
        marker: { size: 9, color: "#d63384", line: { width: 1, color: "#fff" } },
        // Label with the point code used in the export (…L / …R, as in the Top
        // of Girder export), so a point on the plan can be matched to its row.
        // Left labels go above and right labels below so the pairs stay legible.
        text: selectedPoints.map((point) => point.code),
        textposition: selectedPoints.map((point) => (point.side === "L" ? "top center" : "bottom center")),
        textfont: { size: 10, color: "#212529" },
        name: `Span ${selectedSpan} - Girder ${selectedGirder}`,
        customdata: selectedPoints.map((point) => [...toCustomdata(point), point.code]),
        hovertemplate: "<b>%{customdata[3]}</b><br>" + hover,
      });
    }
  }

  if (!traces.length) {
    Plotly.purge(ui.deckChart);
    if (ui.deckStatus) {
      ui.deckStatus.textContent = "Run the girder calculation and upload a DTM to see the deflected deck.";
    }
    return;
  }

  if (ui.deckStatus) {
    if (deck) {
      ui.deckStatus.textContent =
        `${deck.points.length} deflected deck points (${deck.inside} inside the deflected girder area, ` +
        `${deck.points.length - deck.inside} keeping their original elevation).` +
        (deck.edgePoints?.length
          ? ` ${deck.edgePoints.length} overhang edge points ${deck.overhangOffset} ft beyond the edge of deck.`
          : "");
    } else if (state.dtm) {
      ui.deckStatus.textContent =
        `"${state.dtm.name}" loaded with ${state.dtm.points.length} points. ` +
        "Choose Compute Deflected Deck to apply the girder deflections.";
    } else {
      ui.deckStatus.textContent = "Upload a DTM XML surface and choose Compute Deflected Deck.";
    }
  }

  Plotly.react(
    ui.deckChart,
    traces,
    {
      ...PLOTLY_LAYOUT,
      uirevision: state.planRevision,
      title: "<b>Deflected Deck - Plan View (N/E)</b>",
      images,
      xaxis: {
        title: { text: "Easting (ft)", standoff: 34 },
        tickformat: ".0f",
        exponentformat: "none",
        showexponent: "none",
        tickangle: -45,
        automargin: true,
      },
      yaxis: {
        title: { text: "Northing (ft)", standoff: 14 },
        scaleanchor: "x",
        scaleratio: 1,
        tickformat: ".0f",
        exponentformat: "none",
        showexponent: "none",
        automargin: true,
      },
      margin: { t: 60, r: 25, b: 115, l: 95 },
      paper_bgcolor: "#fcfdff",
      plot_bgcolor: "#fcfdff",
      showlegend: false,
    },
    PLOTLY_CONFIG,
  );
  enableZoomWindow(ui.deckChart);
  enableSquareGrid(ui.deckChart);
  bindChartClick(ui.deckChart, onDeckChartClick);
}

// ---------------------------------------------------------------------------
// Alignment sections
// ---------------------------------------------------------------------------

const MAX_SECTION_STATIONS = 2000;
const SECTION_SAMPLES = 600;

/** Every plan point that belongs to the bridge: girder centerlines and deck outline. */
function bridgePlanPoints() {
  const points = [];
  Object.values(state.girderGeometry).forEach((geometry) => {
    (geometry?.planCenterline ?? []).forEach((point) => points.push(point));
  });
  getDeckOutlineRings().forEach((ring) => ring.forEach((point) => points.push(point)));
  return points;
}

/** Station range of the bridge along an alignment, or the whole alignment if unknown. */
function bridgeStationRange(alignment) {
  let min = Infinity;
  let max = -Infinity;
  let known = 0;
  let alongside = 0;
  bridgePlanPoints().forEach((point) => {
    known += 1;
    const hit = alignment.stationOffsetOf(point.e, point.n);
    // Points past the alignment's ends all pile up on the end station.
    if (!hit || hit.beyondEnds) return;
    alongside += 1;
    if (hit.station < min) min = hit.station;
    if (hit.station > max) max = hit.station;
  });
  // An alignment that only grazes the bridge (e.g. ends at it) gives a
  // sliver of a range; treat it as not running alongside.
  if (!(max - min > 1) || alongside < known * 0.25) {
    return { min: alignment.staStart, max: alignment.staEnd, fromBridge: false, offBridge: known > 0 };
  }
  return { min, max, fromBridge: true };
}

/** Mean distance from the bridge to an alignment, used to pick a sensible default. */
function bridgeDistanceTo(alignment) {
  const points = bridgePlanPoints();
  if (!points.length) return 0;
  const step = Math.max(1, Math.floor(points.length / 200));
  let total = 0;
  let count = 0;
  for (let i = 0; i < points.length; i += step) {
    const hit = alignment.stationOffsetOf(points[i].e, points[i].n);
    if (!hit) continue;
    total += hit.distance;
    count += 1;
  }
  return count ? total / count : Infinity;
}

function readSectionInterval() {
  const value = Number(ui.sectionIntervalInput?.value);
  return Number.isFinite(value) && value > 0 ? value : 10;
}

function showStationWarning(message) {
  ui.sectionStationInput.classList.toggle("is-invalid", Boolean(message));
  ui.sectionStationFeedback.textContent = message;
}

/** Fills the station box and its dropdown list with the interval stations. */
function populateSectionStationList() {
  const input = ui.sectionStationInput;
  const menu = ui.sectionStationMenu;
  const hasStations = state.sectionStations.length > 0;
  [input, ui.sectionStationToggle, ui.sectionPrevBtn, ui.sectionNextBtn].forEach((control) => {
    control.disabled = !hasStations;
  });
  showStationWarning("");
  menu.innerHTML = "";

  if (!hasStations) {
    input.value = "";
    input.placeholder = "(Upload an alignment)";
    return;
  }

  input.placeholder = "e.g. 12+34.50";
  state.sectionStations.forEach((station) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "dropdown-item";
    item.dataset.station = String(station);
    item.textContent = BridgeAlignment.formatStation(station);
    if (Math.abs(station - (state.sectionStation ?? NaN)) < 1e-6) item.classList.add("active");
    const row = document.createElement("li");
    row.appendChild(item);
    menu.appendChild(row);
  });
  const current = state.sectionStation;
  input.value = current === null || current === undefined ? "" : BridgeAlignment.formatStation(current);
}

/** Goes to the station typed in the station box, or warns and stays put. */
function goToTypedStation() {
  const text = ui.sectionStationInput.value.trim();
  const range = state.sectionRange;
  if (!state.alignment || !range) return;
  if (!text) {
    populateSectionStationList();
    return;
  }

  const station = BridgeAlignment.parseStation(text);
  if (station === null) {
    showStationWarning("Enter a station such as 12+34.50 or 1234.50.");
    return;
  }
  // Enter and the change that follows it both land here; draw once.
  if (Math.abs(station - (state.sectionStation ?? NaN)) < 1e-6) {
    populateSectionStationList();
    return;
  }
  if (station < range.min - 1e-6 || station > range.max + 1e-6) {
    showStationWarning(
      `Sta ${BridgeAlignment.formatStation(station)} is out of range: ${range.fromBridge ? "the bridge" : "the alignment"} ` +
        `runs from ${BridgeAlignment.formatStation(range.min)} to ${BridgeAlignment.formatStation(range.max)}.`,
    );
    return;
  }
  showSectionAt(station);
}

function rebuildSectionStations() {
  const alignment = state.alignment;
  if (!alignment) {
    state.sectionStations = [];
    state.sectionRange = null;
    return "";
  }

  const range = bridgeStationRange(alignment);
  state.sectionRange = range;
  let interval = readSectionInterval();
  let note = "";
  if ((range.max - range.min) / interval > MAX_SECTION_STATIONS) {
    interval = Math.ceil((range.max - range.min) / MAX_SECTION_STATIONS);
    note = ` Interval raised to ${interval} ft to keep the station list under ${MAX_SECTION_STATIONS} entries.`;
  }

  // Round stations land on the interval; the bridge's own first and last
  // stations are always included so both ends can be inspected.
  const stations = [range.min];
  for (let station = Math.ceil(range.min / interval) * interval; station < range.max; station += interval) {
    if (station - stations[stations.length - 1] > 1e-6) stations.push(station);
  }
  if (range.max - stations[stations.length - 1] > 1e-6) stations.push(range.max);

  // Keep a station the user typed in, if it is still in range.
  const current = state.sectionStation;
  if (
    current !== null &&
    current !== undefined &&
    current >= range.min - 1e-6 &&
    current <= range.max + 1e-6 &&
    !stations.some((station) => Math.abs(station - current) < 1e-6)
  ) {
    stations.push(current);
    stations.sort((a, b) => a - b);
  }

  state.sectionStations = stations;
  if (!stations.some((station) => Math.abs(station - (current ?? NaN)) < 1e-6)) {
    // Default to the station nearest mid-bridge.
    const middle = (range.min + range.max) / 2;
    state.sectionStation = stations.reduce((best, station) =>
      Math.abs(station - middle) < Math.abs(best - middle) ? station : best,
    );
  }

  if (range.fromBridge) {
    return `Bridge spans stations ${BridgeAlignment.formatStation(range.min)} to ${BridgeAlignment.formatStation(range.max)}.${note}`;
  }
  return range.offBridge
    ? `The bridge is not alongside this alignment, so the whole alignment is listed; choose another alignment.${note}`
    : `Run the calculation or load the DTM to limit stations to the bridge.${note}`;
}

/** Rebuilds the station list and redraws the section; call after any input changes. */
function refreshSections() {
  if (!ui.sectionChart) return;
  state.sectionRangeNote = rebuildSectionStations();
  populateSectionStationList();
  renderSectionChart();
}

function nearestListedStation(station) {
  if (!state.sectionStations.length) return station;
  return state.sectionStations.reduce((best, candidate) =>
    Math.abs(candidate - station) < Math.abs(best - station) ? candidate : best,
  );
}

function showSectionAt(station) {
  if (!state.alignment || !Number.isFinite(station)) return;
  state.sectionStation = station;
  if (!state.sectionStations.some((listed) => Math.abs(listed - station) < 1e-6)) {
    state.sectionStations.push(station);
    state.sectionStations.sort((a, b) => a - b);
  }
  populateSectionStationList();
  renderSectionChart();
  renderDeflectedDeckChart();
}

function stepSection(direction) {
  const stations = state.sectionStations;
  if (!stations.length) return;
  const index = stations.findIndex((station) => Math.abs(station - state.sectionStation) < 1e-6);
  const next = Math.max(0, Math.min(stations.length - 1, (index < 0 ? 0 : index) + direction));
  showSectionAt(stations[next]);
}

/** Section-line crossings of a plan polyline, deduplicated at shared vertices. */
function sectionCrossings(origin, right, polyline) {
  const offsets = [];
  BridgeAlignment.crossingsOf(origin, right, polyline).forEach((hit) => {
    if (!offsets.some((offset) => Math.abs(offset - hit.offset) < 1e-6)) offsets.push(hit.offset);
  });
  return offsets;
}

function buildSection(station) {
  const alignment = state.alignment;
  const frame = alignment.pointAt(station);
  const origin = { e: frame.e, n: frame.n };
  const right = { e: frame.rightE, n: frame.rightN };
  const at = (offset) => ({ e: origin.e + right.e * offset, n: origin.n + right.n * offset });

  const mesh = state.deflectedDeck ? state.isopachMesh : null;
  const tin = state.dtmTin;
  const surface = state.deflectedDeck?.deckSurface ?? null;
  const extendsDeck = Boolean(surface?.extends);

  const girders = [];
  sortedSpans().forEach((span) => {
    sortedGirders(span).forEach((girder) => {
      const centerline = state.girderGeometry[`${span}||${girder}`]?.planCenterline;
      if (!centerline?.length) return;
      sectionCrossings(origin, right, centerline).forEach((offset) => girders.push({ span, girder, offset }));
    });
  });

  const overhangs = [];
  (extendsDeck ? mesh.overhangEdges : []).forEach((edge) => {
    sectionCrossings(origin, right, edge.points).forEach((offset) =>
      overhangs.push({ span: edge.span, girder: edge.girder, offset }),
    );
  });

  const deckEdges = [];
  getDeckOutlineRings().forEach((ring) => {
    if (ring.length >= 3) sectionCrossings(origin, right, ring.concat([ring[0]])).forEach((o) => deckEdges.push(o));
  });

  const featureOffsets = girders
    .map((item) => item.offset)
    .concat(overhangs.map((item) => item.offset), deckEdges);
  if (!featureOffsets.length) return { station, origin, right, empty: true };

  let minOffset = Math.min(...featureOffsets);
  let maxOffset = Math.max(...featureOffsets);
  const pad = Math.max(2, (maxOffset - minOffset) * 0.04);
  minOffset -= pad;
  maxOffset += pad;

  const sampleSurfaces = (offset) => {
    const point = at(offset);
    let deck = null;
    if (surface) deck = surface.sample(point.e, point.n);
    else if (tin) {
      const z = tin.sample(point.e, point.n);
      deck = z === null ? null : { z, extended: false };
    }
    if (!deck) return { offset, originalZ: null, isopach: null, deflectedZ: null, extended: false };

    const meshPoint = deck.meshPoint ?? point;
    const hit = mesh ? mesh.sample(meshPoint.e, meshPoint.n) : null;
    const isopach = mesh ? (hit ? hit.value : 0) : null;
    return {
      offset,
      originalZ: deck.z,
      isopach,
      deflectedZ: mesh ? deck.z + isopach : null,
      extended: deck.extended,
    };
  };

  // Regular samples plus the exact feature offsets, so girder and edge
  // elevations are read at the marker rather than interpolated between samples.
  const offsets = [];
  for (let i = 0; i <= SECTION_SAMPLES; i += 1) {
    offsets.push(minOffset + ((maxOffset - minOffset) * i) / SECTION_SAMPLES);
  }
  featureOffsets.forEach((offset) => offsets.push(offset));
  offsets.sort((a, b) => a - b);

  const profile = offsets.map(sampleSurfaces);
  girders.forEach((item) => Object.assign(item, sampleSurfaces(item.offset), at(item.offset)));
  overhangs.forEach((item) => Object.assign(item, sampleSurfaces(item.offset), at(item.offset)));

  return { station, origin, right, minOffset, maxOffset, profile, girders, overhangs, hasDeflected: Boolean(mesh) };
}

/** Vertical exaggeration typed by the user, or null to fit the section to the view. */
function readVerticalExaggeration() {
  const value = Number(ui.verticalExaggerationInput?.value);
  return String(ui.verticalExaggerationInput?.value ?? "").trim() && Number.isFinite(value) && value > 0
    ? value
    : null;
}

/**
 * Shows the exaggeration the section is drawn at: vertical scale over
 * horizontal scale (screen pixels per ft of elevation / per ft of offset).
 * With the box blank this is the fitted value, so the user sees what
 * "Auto" currently means and can type it in to hold it between stations.
 */
function showSectionExaggeration() {
  const input = ui.verticalExaggerationInput;
  const layout = ui.sectionChart?._fullLayout;
  if (!input || !layout?.xaxis?._length || !layout?.yaxis?._length) return;
  const xSpan = Math.abs(layout.xaxis.range[1] - layout.xaxis.range[0]);
  const ySpan = Math.abs(layout.yaxis.range[1] - layout.yaxis.range[0]);
  if (!(xSpan > 0) || !(ySpan > 0)) return;
  const current = layout.yaxis._length / ySpan / (layout.xaxis._length / xSpan);
  input.placeholder = `Auto (${current >= 10 ? current.toFixed(0) : current.toFixed(1)})`;
}

function renderSectionChart() {
  if (!ui.sectionChart) return;

  const setStatus = (text) => {
    ui.sectionStatus.textContent = text;
  };

  state.section = null;
  if (!state.alignment) {
    Plotly.purge(ui.sectionChart);
    setStatus("Upload a civil alignment (LandXML) to draw sections across the deck.");
    return;
  }
  if (!state.dtmTin?.triangleCount) {
    Plotly.purge(ui.sectionChart);
    setStatus(
      state.dtm
        ? "The DTM has no TIN faces, so sections cannot be sampled from it."
        : "Upload the top-of-deck DTM to draw sections.",
    );
    return;
  }

  const station = state.sectionStation;
  if (!Number.isFinite(station)) {
    Plotly.purge(ui.sectionChart);
    setStatus("Choose a station.");
    return;
  }

  const section = buildSection(station);
  const exaggeration = readVerticalExaggeration();
  const stationText = BridgeAlignment.formatStation(station);
  if (section.empty) {
    Plotly.purge(ui.sectionChart);
    setStatus(`Station ${stationText} does not cross the deck or any girder.`);
    return;
  }
  state.section = section;

  const x = section.profile.map((point) => point.offset);
  const profile = section.profile;
  // Split each surface into the part read from the DTM (solid) and the part
  // carried out to the screed line (dashed), at the cross slope or level. The
  // dashed part repeats its neighbouring DTM sample so the two lines join.
  const touchesExtended = (i) => profile[i - 1]?.extended || profile[i + 1]?.extended;
  const modelPart = (value) => profile.map((point) => (point.extended ? null : value(point)));
  const extendedPart = (value) =>
    profile.map((point, i) => (point.extended || (value(point) !== null && touchesExtended(i)) ? value(point) : null));
  const hasExtended = profile.some((point) => point.extended);
  const deflectionIn = profile.map((point) => (point.isopach === null ? null : point.isopach * 12));

  const traces = [
    {
      x,
      y: modelPart((point) => point.originalZ),
      mode: "lines",
      line: { width: 2.5, color: "#1b5ba3" },
      name: "Original DTM",
      legendgroup: "original",
      hovertemplate: "%{y:.3f} ft<extra>Original</extra>",
    },
  ];
  if (hasExtended) {
    traces.push({
      x,
      y: extendedPart((point) => point.originalZ),
      mode: "lines",
      line: { width: 2.5, color: "#1b5ba3", dash: "dash" },
      name: `Original, ${extensionLabel()}`,
      legendgroup: "original",
      hovertemplate: "%{y:.3f} ft<extra>Original (extended)</extra>",
    });
  }

  if (section.hasDeflected) {
    traces.push({
      x,
      y: modelPart((point) => point.deflectedZ),
      customdata: deflectionIn,
      mode: "lines",
      line: { width: 2.5, color: "#dc3545" },
      name: "Deflected",
      legendgroup: "deflected",
      hovertemplate: "%{y:.3f} ft (deflection %{customdata:.3f} in)<extra>Deflected</extra>",
    });
    if (hasExtended) {
      traces.push({
        x,
        y: extendedPart((point) => point.deflectedZ),
        customdata: deflectionIn,
        mode: "lines",
        line: { width: 2.5, color: "#dc3545", dash: "dash" },
        name: `Deflected, ${extensionLabel()}`,
        legendgroup: "deflected",
        hovertemplate: "%{y:.3f} ft (deflection %{customdata:.3f} in)<extra>Deflected (extended)</extra>",
      });
    }
  }

  const spansCrossed = new Set(section.girders.map((item) => item.span));
  const girderLabel = (item) => (spansCrossed.size > 1 ? `${item.span}-G${item.girder}` : `G${item.girder}`);
  const markerZ = (item) => (section.hasDeflected ? item.deflectedZ : item.originalZ);

  const girdersWithZ = section.girders.filter((item) => markerZ(item) !== null);
  if (girdersWithZ.length) {
    traces.push({
      x: girdersWithZ.map((item) => item.offset),
      y: girdersWithZ.map(markerZ),
      mode: "markers",
      marker: { size: 9, color: "#212529", symbol: "triangle-down", line: { width: 1, color: "#fff" } },
      name: "Girders",
      customdata: girdersWithZ.map((item) => [
        item.span,
        item.girder,
        item.originalZ,
        item.isopach === null ? 0 : item.isopach * 12,
        item.n,
        item.e,
      ]),
      hovertemplate:
        "<b>Span %{customdata[0]} - Girder %{customdata[1]}</b><br>Offset %{x:.3f} ft<br>" +
        "N %{customdata[4]:.3f}  E %{customdata[5]:.3f}<br>DTM %{customdata[2]:.3f} ft<br>" +
        (section.hasDeflected ? "Deflection %{customdata[3]:.3f} in<br><b>Deflected %{y:.3f} ft</b>" : "") +
        "<extra></extra>",
    });
  }

  const overhangsWithZ = section.overhangs.filter((item) => markerZ(item) !== null);
  if (overhangsWithZ.length) {
    traces.push({
      x: overhangsWithZ.map((item) => item.offset),
      y: overhangsWithZ.map(markerZ),
      mode: "markers",
      marker: { size: 9, color: "#fd7e14", symbol: "diamond", line: { width: 1, color: "#fff" } },
      name: "Overhang edges",
      customdata: overhangsWithZ.map((item) => [
        item.span,
        item.girder,
        item.originalZ,
        item.isopach === null ? 0 : item.isopach * 12,
        item.extended ? extensionLabel() : "from DTM",
      ]),
      hovertemplate:
        "<b>Overhang edge (Span %{customdata[0]}, Girder %{customdata[1]})</b><br>Offset %{x:.3f} ft<br>" +
        "Deck %{customdata[2]:.3f} ft (%{customdata[4]})<br>Deflection %{customdata[3]:.3f} in<br>" +
        "<b>Deflected %{y:.3f} ft</b><extra></extra>",
    });
  }

  const shapes = [];
  const annotations = [];
  const verticalLine = (offset, color, dash) =>
    shapes.push({
      type: "line",
      xref: "x",
      yref: "paper",
      x0: offset,
      x1: offset,
      y0: 0,
      y1: 1,
      line: { color, width: 1, dash },
    });
  const topLabel = (offset, text, color) =>
    annotations.push({
      x: offset,
      y: 1,
      xref: "x",
      yref: "paper",
      yanchor: "bottom",
      text,
      showarrow: false,
      font: { size: 10, color },
    });

  section.girders.forEach((item) => {
    verticalLine(item.offset, "#6c757d", "dot");
    topLabel(item.offset, girderLabel(item), "#343a40");
  });
  section.overhangs.forEach((item) => {
    verticalLine(item.offset, "#fd7e14", "dash");
    topLabel(item.offset, "OH", "#c35a00");
  });
  if (section.minOffset <= 0 && section.maxOffset >= 0) {
    verticalLine(0, "#198754", "dashdot");
    topLabel(0, "CL", "#198754");
  }

  Plotly.newPlot(
    ui.sectionChart,
    traces,
    {
      ...PLOTLY_LAYOUT,
      title: { text: `<b>Section at Sta ${stationText}</b> - ${state.alignment.name}`, y: 0.97 },
      xaxis: {
        title: { text: "Offset from alignment (ft) - left negative, right positive" },
        range: [section.minOffset, section.maxOffset],
        zeroline: false,
      },
      yaxis: {
        title: { text: "Elevation (ft)" },
        tickformat: ".2f",
        automargin: true,
        // A set exaggeration locks 1 ft of elevation to `exaggeration` ft of
        // offset on screen; blank lets the section fill the view.
        ...(exaggeration ? { scaleanchor: "x", scaleratio: exaggeration } : {}),
      },
      hovermode: "closest",
      shapes,
      annotations,
      margin: { t: 90, r: 25, b: 120, l: 80 },
      paper_bgcolor: "#fcfdff",
      plot_bgcolor: "#fcfdff",
      showlegend: true,
      // Pinned to the bottom of the chart (not a fraction of the plot height),
      // so it stays clear of the axis title however short the chart is.
      legend: { orientation: "h", x: 0, xref: "container", y: 0, yref: "container", yanchor: "bottom" },
    },
    PLOTLY_CONFIG,
  );
  enableZoomWindow(ui.sectionChart);
  showSectionExaggeration();
  ui.sectionChart.on("plotly_relayout", showSectionExaggeration);
  ui.sectionChart.on("plotly_update", showSectionExaggeration); // zoom window

  const deflections = section.girders.map((item) => item.isopach).filter((value) => value !== null);
  const parts = [
    `Sta ${stationText}: ${section.girders.length} girder crossing(s), ${section.overhangs.length} overhang edge(s).`,
  ];
  if (section.hasDeflected && deflections.length) {
    const largest = deflections.reduce((best, value) => (Math.abs(value) > Math.abs(best) ? value : best));
    parts.push(`Largest girder deflection here: ${(largest * 12).toFixed(3)} in.`);
  } else if (!section.hasDeflected) {
    parts.push("Compute the deflected deck to add the deflected surface and overhang edges.");
  }
  if (state.sectionRangeNote) parts.push(state.sectionRangeNote);
  setStatus(parts.join(" "));
}

/** Alignment (clipped near the bridge) and the current section line, for the deck plan. */
function alignmentPlanTraces() {
  const alignment = state.alignment;
  if (!alignment) return [];

  const range = bridgeStationRange(alignment);
  const margin = range.fromBridge ? Math.max(20, (range.max - range.min) * 0.1) : 0;
  const from = Math.max(alignment.staStart, range.min - margin);
  const to = Math.min(alignment.staEnd, range.max + margin);

  // Sample the exact geometry across the clipped range rather than filtering
  // the densified vertices: a long straight <Line> has only its two end
  // vertices, which both fall outside the range when it runs past the bridge.
  // Regular samples also give every point a station to click on.
  const count = Math.max(2, Math.min(1000, Math.ceil((to - from) / 2) + 1));
  const shown = [];
  for (let i = 0; i < count && to > from; i += 1) {
    const station = from + ((to - from) * i) / (count - 1);
    const frame = alignment.pointAt(station);
    shown.push({ e: frame.e, n: frame.n, station });
  }
  const traces = [];
  if (shown.length >= 2) {
    traces.push({
      x: shown.map((point) => point.e),
      y: shown.map((point) => point.n),
      mode: "lines",
      line: { width: 2, color: "#198754", dash: "dashdot" },
      name: alignment.name,
      customdata: shown.map((point) => [point.station]),
      hovertemplate: `${alignment.name}<br>Sta %{customdata[0]:.2f}<br>Click to cut a section<extra></extra>`,
    });
  }

  const section = state.section;
  if (section && !section.empty) {
    const ends = [section.minOffset, section.maxOffset].map((offset) => ({
      e: section.origin.e + section.right.e * offset,
      n: section.origin.n + section.right.n * offset,
    }));
    traces.push({
      x: ends.map((point) => point.e),
      y: ends.map((point) => point.n),
      mode: "lines+text",
      line: { width: 3, color: "#0dcaf0" },
      text: ["", `Sta ${BridgeAlignment.formatStation(section.station)}`],
      textposition: "middle right",
      textfont: { size: 11, color: "#055160" },
      name: "Section line",
      hoverinfo: "skip",
    });
  }
  return traces;
}

async function loadAlignmentFile() {
  const [file] = ui.alignmentFileInput.files;
  state.alignments = [];
  state.alignment = null;
  state.sectionStation = null;

  if (!file) {
    ui.alignmentUploadStatus.textContent = "";
    populateAlignmentSelect();
    refreshSections();
    renderDeflectedDeckChart();
    return;
  }

  const { alignments, errors } = BridgeAlignment.parseAlignments(await readTextFile(file));
  state.alignments = alignments;
  errors.forEach((message) => logLine(`Alignment WARNING: ${message}`));

  // Default to the alignment that runs closest to the bridge.
  let bestIndex = 0;
  if (alignments.length > 1) {
    let bestDistance = Infinity;
    alignments.forEach((alignment, index) => {
      const distance = bridgeDistanceTo(alignment);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = index;
      }
    });
  }

  populateAlignmentSelect(bestIndex);
  selectAlignment(bestIndex);
  ui.alignmentUploadStatus.textContent = `Loaded ${alignments.length} alignment(s) from "${file.name}".`;
}

function populateAlignmentSelect(selectedIndex = 0) {
  const select = ui.alignmentSelect;
  if (!state.alignments.length) {
    select.innerHTML = '<option value="">(Upload an alignment)</option>';
    select.disabled = true;
    return;
  }
  select.disabled = false;
  select.innerHTML = "";
  state.alignments.forEach((alignment, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent =
      `${alignment.name} (Sta ${BridgeAlignment.formatStation(alignment.staStart)} to ` +
      `${BridgeAlignment.formatStation(alignment.staEnd)})`;
    select.appendChild(option);
  });
  select.value = String(selectedIndex);
}

function selectAlignment(index) {
  const alignment = state.alignments[index];
  if (!alignment) return;
  state.alignment = alignment;
  state.sectionStation = null;

  logLine(
    `Alignment: "${alignment.name}" - ${alignment.elementCount} element(s), stations ` +
      `${BridgeAlignment.formatStation(alignment.staStart)} to ${BridgeAlignment.formatStation(alignment.staEnd)}.`,
  );
  alignment.warnings.forEach((warning) => logLine(`Alignment WARNING (${alignment.name}): ${warning}`));

  refreshSections();
  renderDeflectedDeckChart();
}

function runCalculation() {
  if (!state.sourceRows.length) {
    window.alert("Please upload the input Excel file.");
    return;
  }

  const intervals = parseNumber(ui.intervalsInput.value, Number.NaN);
  if (!Number.isInteger(intervals) || intervals < 1 || intervals > 250) {
    window.alert("Intervals must be an integer between 1 and 250.");
    return;
  }

  state.logs = [];
  state.profiles = {};
  state.spanToGirders = {};
  state.girderGeometry = {};
  // Deflections are about to change, so any deck computed from them is stale.
  state.isopachMesh = null;
  state.deflectedDeck = null;
  state.planRevision += 1;

  const output = [["N", "E", "Elevation (ft)", "Description", "Deflection (ft)", "Camber (ft)"]];

  const total = state.sourceRows.length;
  for (let rowIndex = 0; rowIndex < total; rowIndex += 1) {
    const row = normalizeRow(state.sourceRows[rowIndex]);
    state.sourceRows[rowIndex] = row;

    try {
      const result = buildGirderPoints(row, intervals);
      output.push(...result.rows);

      const profileKey = `${result.spanDisplay}||${result.girderDisplay}`;
      state.profiles[profileKey] = result.graphPoints;
      state.girderGeometry[profileKey] = {
        support1N: result.support1N,
        support1E: result.support1E,
        support2N: result.support2N,
        support2E: result.support2E,
        planCenterline: result.planCenterline,
        planEdges: result.planEdges,
      };

      if (!state.spanToGirders[result.spanDisplay]) {
        state.spanToGirders[result.spanDisplay] = new Set();
      }
      state.spanToGirders[result.spanDisplay].add(result.girderDisplay);

      state.logs.push(
        `Row ${rowIndex + 2}: Span ${result.spanDisplay}, Girder ${result.girderDisplay}. A_def=${result.aDefIn.toFixed(3)} in, A_camber=${result.aCamberIn.toFixed(3)} in, centerline radius=${result.centerlineRadius.toFixed(3)} ft.`,
      );
    } catch (error) {
      state.logs.push(`Row ${rowIndex + 2}: ERROR - ${error.message}`);
    }

    const pct = Math.round(((rowIndex + 1) / total) * 100);
    setProgress(pct, `Processed row ${rowIndex + 1} of ${total}`);
  }

  state.topOfGirderPoints = output;
  ui.logOutput.textContent = state.logs.join("\n");
  populateGraphSelectors();
  renderProfileChart();
  renderPlanChart();
  setProgress(100, "Calculation complete");
}

function exportTopOfGirderPoints() {
  if (state.topOfGirderPoints.length <= 1) {
    window.alert("Please run the calculation first.");
    return;
  }

  exportRowsAsWorkbook(state.topOfGirderPoints, "Top of girder.xlsx");
  logLine(`Export: wrote ${state.topOfGirderPoints.length - 1} top-of-girder points to "Top of girder.xlsx".`);
}

function exportTopOfDeckDeflected() {
  if (!state.topOfGirderPoints.length) {
    window.alert("Please calculate the top-of-girder points first.");
    return;
  }

  if (state.dtm && !state.deflectedDeck && !computeDeflectedDeck()) {
    return;
  }

  if (state.deflectedDeck) {
    const girderPoints = state.deflectedDeck.girderPoints ?? {};
    const edgePoints = state.deflectedDeck.edgePoints ?? [];
    const rows = [
      ["N", "E", "Deflected Elevation (ft)", "Description", "Deck Elevation (ft)", "Deflection (ft)", "Note"],
    ];

    let screedRows = 0;
    sortedSpans().forEach((span) => {
      sortedGirders(span).forEach((girder) => {
        (girderPoints[`${span}||${girder}`] ?? []).forEach((point) => {
          rows.push([
            point.n,
            point.e,
            point.deflectedZ,
            point.code,
            point.originalZ,
            point.isopach,
            "",
          ]);
        });
      });

      // Screed points along the overhang edges, after the span's girders.
      edgePoints
        .filter((point) => point.span === span)
        .forEach((point) => {
          screedRows += 1;
          rows.push([
            point.n,
            point.e,
            point.deflectedZ,
            `${formatSpan(span)}${formatGirder(point.girder)}${formatInterval(point.interval)}OH`,
            point.originalZ,
            point.isopach,
            !point.extended
              ? "Overhang edge; deck from DTM"
              : state.deflectedDeck.overhangSlope === "level"
                ? "Overhang edge; deck held level from the edge of deck past the DTM"
                : `Overhang edge; deck carried at ${(point.crossSlope * 100).toFixed(2)}% cross slope past the DTM`,
          ]);
        });
    });

    if (rows.length === 1) {
      window.alert(
        "No deflected points could be sampled along the girders. The DTM may not have TIN faces, " +
          "or it may not cover the girder lines.",
      );
      return;
    }

    exportRowsAsWorkbook(rows, "ToD Deflected.xlsx");
    logLine(
      `Export: wrote ${rows.length - 1 - screedRows} deflected deck points at girder intervals` +
        (screedRows ? ` and ${screedRows} overhang (screed) points` : "") +
        ' to "ToD Deflected.xlsx".',
    );
    return;
  }

  const projectedRows = [["N", "E", "Elevation (ft)", "Description"]];
  for (let i = 1; i < state.topOfGirderPoints.length; i += 1) {
    const [n, e, elevation, desc] = state.topOfGirderPoints[i];
    projectedRows.push([n, e, elevation, desc]);
  }

  exportRowsAsWorkbook(projectedRows, "ToD Deflected.xlsx");
  logLine(
    "Projection note: No DTM XML provided. Export used computed top-of-girder elevations only (no deck surface applied).",
  );
}

// Plan grid (ft) used to densify the deck TIN so the deflection curve is
// captured between the DTM's own vertices.
const SURFACE_CELL_SIZE = 5;

function exportDeflectedSurfaceXml() {
  if (!state.topOfGirderPoints.length) {
    window.alert("Please calculate the top-of-girder points first (Girder Calcs tab).");
    return;
  }
  if (!state.dtm) {
    window.alert("Please upload the top-of-deck DTM XML surface first.");
    return;
  }
  if (!state.deflectedDeck && !computeDeflectedDeck()) return;
  if (!state.dtm.faces.length) {
    window.alert("The DTM has no TIN faces, so a deflected surface cannot be built from it.");
    return;
  }

  const mesh = state.isopachMesh;
  const tin = state.dtmTin;
  const { overhangOffset, overhangSlope } = state.deflectedDeck;
  const surface = BridgeSurfaceExport.buildDeflectedSurface({
    points: state.dtm.points,
    faces: state.dtm.faces,
    isopachAt: (e, n) => mesh.sample(e, n)?.value ?? 0,
    deckZ: (e, n) => tin.sample(e, n),
    cellSize: SURFACE_CELL_SIZE,
    overhangOffset,
    overhangSlope,
    // The exterior girders, so the deck's side edges can be found locally.
    fascias: mesh.overhangEdges,
    crossSlopeRun: CROSS_SLOPE_RUN,
  });

  const name = `${state.dtm.name} - Deflected`;
  const description =
    overhangOffset === null
      ? "Top of deck plus girder deflection"
      : `Top of deck plus girder deflection, extended ${overhangOffset} ft beyond the edge of deck, ` +
        `${describeOverhangSlope(overhangSlope)}; the edge of deck is a break line`;
  const xml = BridgeSurfaceExport.toLandXml(surface, { name, description, unitsXml: state.dtmUnitsXml });

  const url = URL.createObjectURL(new Blob([xml], { type: "application/xml" }));
  triggerDownload(url, "ToD Deflected Surface.xml");
  logLine(
    `Export: wrote surface "${name}" to "ToD Deflected Surface.xml" - ${surface.vertices.length} points, ` +
      `${surface.faces.length} faces (deck densified on a ${SURFACE_CELL_SIZE} ft grid` +
      (surface.stripFaces ? `, plus ${surface.stripFaces} faces out to the overhang edge).` : ")."),
  );
  if (surface.breaklines.length) {
    logLine(
      `Export: the edge of deck is written as ${surface.breaklines.length} break line(s); the overhang grade ` +
        "starts there.",
    );
  }
  if (overhangOffset !== null && !surface.stripFaces) {
    logLine("Export WARNING: no deck side edges were found to extend, so the surface stops at the DTM edge.");
  }
  return { surface, xml };
}

function downloadLog() {
  const blob = new Blob([ui.logOutput.textContent || "No log entries yet."], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  triggerDownload(url, "Log.txt");
}

renderSourceGrid();

ui.tabDataBtn.addEventListener("click", () => activateTab("data"));
ui.tabGraphsBtn.addEventListener("click", () => activateTab("graphs"));
ui.tabExportBtn.addEventListener("click", () => activateTab("export"));

document.getElementById("downloadTemplateBtn").addEventListener("click", downloadTemplate);
document.getElementById("calculateBtn").addEventListener("click", runCalculation);
document.getElementById("exportGirderBtn").addEventListener("click", exportTopOfGirderPoints);
document.getElementById("projectBtn").addEventListener("click", exportTopOfDeckDeflected);
document.getElementById("exportSurfaceBtn").addEventListener("click", exportDeflectedSurfaceXml);
document.getElementById("downloadLogBtn").addEventListener("click", downloadLog);

ui.fileInput.addEventListener("change", async () => {
  try {
    await loadSourceRows();
  } catch (error) {
    state.sourceRows = [];
    renderSourceGrid();
    ui.uploadStatus.textContent = `Error loading spreadsheet: ${error.message}`;
  }
});

ui.dtmFileInput.addEventListener("change", async () => {
  try {
    await loadDtmSurface();
  } catch (error) {
    state.dtm = null;
    state.dtmTin = null;
    state.dtmBoundary = null;
    state.deflectedDeck = null;
    state.planRevision += 1;
    ui.dtmUploadStatus.textContent = `Error loading DTM: ${error.message}`;
    logLine(`DTM ERROR: ${error.message}`);
    refreshSections();
    renderDeflectedDeckChart();
  }
});

if (ui.overhangInput) {
  // A new overhang only reshapes the isopach edges, so recompute straight
  // away when a deck is already showing.
  ui.overhangInput.addEventListener("change", () => {
    if (state.deflectedDeck) computeDeflectedDeck();
  });
}

if (ui.overhangSlopeSelect) {
  // Same for the overhang grade, which only changes the extension.
  ui.overhangSlopeSelect.addEventListener("change", () => {
    if (state.deflectedDeck) computeDeflectedDeck();
  });
}

if (ui.alignmentFileInput) {
  ui.alignmentFileInput.addEventListener("change", async () => {
    try {
      await loadAlignmentFile();
    } catch (error) {
      state.alignments = [];
      state.alignment = null;
      populateAlignmentSelect();
      ui.alignmentUploadStatus.textContent = `Error loading alignment: ${error.message}`;
      logLine(`Alignment ERROR: ${error.message}`);
      refreshSections();
      renderDeflectedDeckChart();
    }
  });

  ui.alignmentSelect.addEventListener("change", () => selectAlignment(Number(ui.alignmentSelect.value)));
  ui.sectionIntervalInput.addEventListener("change", () => {
    refreshSections();
    renderDeflectedDeckChart();
  });
  ui.verticalExaggerationInput.addEventListener("change", renderSectionChart);
  ui.sectionPrevBtn.addEventListener("click", () => stepSection(-1));
  ui.sectionNextBtn.addEventListener("click", () => stepSection(1));

  // The station box takes a typed station (Enter, or leaving the box) or a
  // pick from its dropdown list of interval stations.
  ui.sectionStationInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      goToTypedStation();
    } else if (event.key === "Escape") {
      populateSectionStationList(); // back to the current station
    }
  });
  ui.sectionStationInput.addEventListener("change", goToTypedStation);
  ui.sectionStationInput.addEventListener("focus", () => ui.sectionStationInput.select());
  ui.sectionStationMenu.addEventListener("click", (event) => {
    const item = event.target.closest("[data-station]");
    if (item) showSectionAt(Number(item.dataset.station));
  });
  // Open the list at the current station rather than at the top.
  ui.sectionStationToggle.addEventListener("shown.bs.dropdown", () => {
    ui.sectionStationMenu.querySelector(".active")?.scrollIntoView({ block: "center" });
  });
}

ui.graphSpanSelect.addEventListener("change", () => {
  populateGirderSelect(ui.graphSpanSelect.value, ui.graphGirderSelect);
  renderProfileChart();
  renderPlanChart();
});

ui.graphGirderSelect.addEventListener("change", () => {
  renderProfileChart();
  renderPlanChart();
});

function onPlanChartClick(event) {
  if (event?.event?.button !== 0) return;
  const payload = event?.points?.[0]?.customdata;
  if (!payload) return;
  const [span, girder] = payload;
  if (ui.graphSpanSelect.value !== span) {
    ui.graphSpanSelect.value = span;
    populateGirderSelect(span, ui.graphGirderSelect);
  }
  ui.graphGirderSelect.value = girder;
  renderProfileChart();
  renderPlanChart();
}

if (ui.deckSpanSelect) {
  ui.deckSpanSelect.addEventListener("change", () => {
    populateGirderSelect(ui.deckSpanSelect.value, ui.deckGirderSelect);
    renderDeflectedDeckChart();
  });
}

if (ui.deckGirderSelect) {
  ui.deckGirderSelect.addEventListener("change", renderDeflectedDeckChart);
}

if (ui.planSurfaceSelect) {
  ui.planSurfaceSelect.addEventListener("change", renderDeflectedDeckChart);
}

if (ui.contoursToggle) {
  ui.contoursToggle.addEventListener("change", renderDeflectedDeckChart);
  ui.contourIntervalInput.addEventListener("change", renderDeflectedDeckChart);
}

function onDeckChartClick(event) {
  if (event?.event?.button !== 0) return;
  const payload = event?.points?.[0]?.customdata;
  // The alignment trace carries [station]; girders carry [span, girder].
  if (Array.isArray(payload) && payload.length === 1 && Number.isFinite(payload[0])) {
    showSectionAt(nearestListedStation(payload[0]));
    return;
  }
  if (!payload || payload.length !== 2) return;
  const [span, girder] = payload;
  if (ui.deckSpanSelect.value !== span) {
    ui.deckSpanSelect.value = span;
    populateGirderSelect(span, ui.deckGirderSelect);
  }
  ui.deckGirderSelect.value = girder;
  renderDeflectedDeckChart();
}

const computeDeckBtn = document.getElementById("computeDeckBtn");
if (computeDeckBtn) {
  computeDeckBtn.addEventListener("click", () => {
    computeDeflectedDeck();
  });
}

setProgress(0, "Waiting for input");
refreshSections();

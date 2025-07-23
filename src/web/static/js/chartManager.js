// ChartManager - Unified chart wrapper for ReViision analytics
// Dependencies: Chart.js v3+, chartjs-plugin-zoom, chartjs-plugin-datalabels, FileSaver.js
// This module registers the plugins once and exposes convenient helper methods for
// switching chart type, toggling datasets, zoom controls, and exporting the chart
// data/visualisation in PNG, CSV, or JSON formats.

(function(global) {
    "use strict";

    if (!global.Chart) {
        console.error("Chart.js not found – ChartManager requires Chart.js v3+");
        return;
    }

    // Attempt to register plugins if they exist in the global scope
    const maybeRegister = (pluginGlobalName) => {
        if (global[pluginGlobalName]) {
            try {
                global.Chart.register(global[pluginGlobalName]);
            } catch (err) {
                // Ignore if already registered
            }
        }
    };

    maybeRegister("chartjsZoom");            // chartjs-plugin-zoom attaches itself under this name
    maybeRegister("ChartDataLabels");        // chartjs-plugin-datalabels
    maybeRegister("ChartZoom"); // alternative global var name for zoom plugin

    const saveAs = global.saveAs; // Provided by FileSaver.js

    class ChartManager {
        constructor(canvasCtx, config, options = {}) {
            // Create underlying Chart.js instance
            this.canvas = canvasCtx;
            // Deep-clone scales for restoration when switching types
            this._originalScales = config.options && config.options.scales ? JSON.parse(JSON.stringify(config.options.scales)) : undefined;

            // Ensure zoom options are present by default
            const zoomDefaults = {
                zoom: {
                    wheel: { enabled: true },
                    pinch: { enabled: true },
                    mode: 'xy'
                },
                pan: { enabled: true, mode: 'xy' }
            };
            config.options = config.options || {};
            config.options.plugins = config.options.plugins || {};
            config.options.plugins.zoom = Object.assign({}, zoomDefaults, config.options.plugins.zoom || {});

            this.chart = new global.Chart(canvasCtx, config);
            // Merge user-provided options (e.g., default filename stub, id)
            this.options = Object.assign({ filename: "chart", toolbar: true }, options);

            if (this.options.toolbar) {
                this._buildToolbar();
            }
        }

        // ---------- Core helpers ---------- //
        switchType(newType) {
            if (this.chart.config.type === newType) return;
            this.chart.config.type = newType;

            // Axes handling: remove for radial charts, restore for cartesian
            const radialTypes = ['doughnut', 'pie', 'polarArea', 'radar'];
            if (radialTypes.includes(newType)) {
                this.chart.options.scales = {};
            } else {
                if (this._originalScales) {
                    this.chart.options.scales = JSON.parse(JSON.stringify(this._originalScales));
                }
            }

            this.chart.update();
        }

        toggleDataset(dsIndex) {
            // Chart.js v3+ has built-in helpers
            if (this.chart.isDatasetVisible(dsIndex)) {
                this.chart.hide(dsIndex);
            } else {
                this.chart.show(dsIndex);
            }
        }

        resetZoom() {
            if (this.chart.resetZoom) {
                this.chart.resetZoom();
            }
        }

        resize() {
            this.chart.resize();
        }

        // ---------- Export helpers ---------- //
        exportPNG(filename) {
            const name = filename || `${this.options.filename}.png`;
            // Ensure the chart is rendered at current size
            this.chart.resize();
            this.canvas.toBlob((blob) => {
                if (blob && saveAs) {
                    saveAs(blob, name);
                } else {
                    console.error("FileSaver.js saveAs function missing or canvas blob failed");
                }
            }, "image/png", 1.0);
        }

        exportCSV(filename) {
            const name = filename || `${this.options.filename}.csv`;
            const { labels = [] } = this.chart.data;
            const datasets = this.chart.data.datasets || [];

            // Build CSV rows (header + data rows)
            const rows = [];
            const header = ["Label", ...datasets.map((d) => sanitizeCSV(d.label))];
            rows.push(header.join(","));

            labels.forEach((lbl, idx) => {
                const row = [sanitizeCSV(lbl)];
                datasets.forEach((ds) => {
                    row.push(ds.data[idx] != null ? ds.data[idx] : "");
                });
                rows.push(row.join(","));
            });

            const csvBlob = new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8" });
            saveAs(csvBlob, name);
        }

        exportJSON(filename) {
            const name = filename || `${this.options.filename}.json`;
            const json = JSON.stringify({ labels: this.chart.data.labels, datasets: this.chart.data.datasets }, null, 2);
            const blob = new Blob([json], { type: "application/json;charset=utf-8" });
            saveAs(blob, name);
        }

        destroy() {
            this.chart.destroy();
        }

        // Build toolbar UI and attach event listeners
        _buildToolbar() {
            const container = this.canvas.closest('.chart-container') || this.canvas.parentElement;
            if (!container) return;
            // Ensure container is positioned for absolute toolbar
            if (getComputedStyle(container).position === 'static') {
                container.style.position = 'relative';
            }
            // Avoid duplicates
            if (container.querySelector('.chart-toolbar')) return;

            // Inject toolbar CSS once
            if (!document.getElementById('chart-toolbar-style')) {
                const style = document.createElement('style');
                style.id = 'chart-toolbar-style';
                style.textContent = `
                     .chart-toolbar { position: absolute; top: 6px; right: 10px; z-index: 40; display: flex; gap: 4px; flex-wrap: nowrap; background: rgba(255,255,255,0.85); border-radius: 4px; padding: 2px 4px; box-shadow: 0 0 4px rgba(0,0,0,0.1); }
                     .chart-toolbar .btn { padding: 2px 6px; font-size: 0.75rem; line-height: 1; }
                     .chart-container { overflow: visible; } /* prevent tooltip clipping */
                     .card-header { overflow: visible; }
                     .tooltip .tooltip-text { left: auto; right: 110%; top: 50%; transform: translateY(-50%); white-space: normal; max-width: 220px; }
                 `;
                document.head.appendChild(style);
            }

            const TYPES = ['bar', 'line', 'doughnut', 'pie', 'radar'];

            const toolbar = document.createElement('div');
            toolbar.className = 'chart-toolbar text-end mt-1';
            toolbar.innerHTML = `
                <div class="btn-group me-1">
                    <button class="btn btn-sm btn-outline-secondary dropdown-toggle" data-bs-toggle="dropdown" aria-expanded="false" title="Change Chart Type"><i class="fas fa-random"></i></button>
                    <ul class="dropdown-menu dropdown-menu-end">
                        ${TYPES.map(t => `<li><a class="dropdown-item" data-type="${t}" href="#">${t.charAt(0).toUpperCase() + t.slice(1)}</a></li>`).join('')}
                    </ul>
                </div>
                <button class="btn btn-sm btn-outline-secondary me-1" data-action="reset-zoom" title="Reset Zoom"><i class="fas fa-search-minus"></i></button>
                <button class="btn btn-sm btn-outline-secondary me-1" data-action="png" title="Download PNG"><i class="fas fa-image"></i></button>
                <button class="btn btn-sm btn-outline-secondary me-1" data-action="csv" title="Download CSV"><i class="fas fa-file-csv"></i></button>
                <button class="btn btn-sm btn-outline-secondary" data-action="json" title="Download JSON"><i class="fas fa-code"></i></button>
            `;

            container.appendChild(toolbar);

            // Event delegation for toolbar actions
            toolbar.addEventListener('click', (ev) => {
                const typeItem = ev.target.closest('[data-type]');
                if (typeItem) {
                    ev.preventDefault();
                    const newType = typeItem.getAttribute('data-type');
                    this.switchType(newType);
                    return;
                }

                const btn = ev.target.closest('[data-action]');
                if (!btn) return;
                const action = btn.getAttribute('data-action');
                if (action === 'png') {
                    this.exportPNG();
                } else if (action === 'csv') {
                    this.exportCSV();
                } else if (action === 'json') {
                    this.exportJSON();
                } else if (action === 'reset-zoom') {
                    this.resetZoom();
                }
            });
        }
    }

    function sanitizeCSV(value) {
        // Escape quotes / commas
        if (typeof value === "string" && (value.includes(",") || value.includes("\""))) {
            return `"${value.replace(/"/g, '""')}"`;
        }
        return value;
    }

    // Expose globally
    global.ChartManager = ChartManager;
})(window); 
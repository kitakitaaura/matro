//! Lightweight, zero-dependency scientific plotting model with subplots, heatmaps (`imagesc`),
//! legends, log scales, explicit axis limits, and clean light-theme SVG / JSON output.

use crate::Matrix;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlotKind {
    Line,
    Scatter,
    Bar,
    Stem,
    ErrorBar,
    Fill,
}

impl PlotKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Line => "line",
            Self::Scatter => "scatter",
            Self::Bar => "bar",
            Self::Stem => "stem",
            Self::ErrorBar => "errorbar",
            Self::Fill => "fill",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContourData {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>, // row-major: y.len() x x.len()
    pub levels: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlotSeries {
    pub kind: PlotKind,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub err: Vec<f64>,
    pub color: String,
    pub dashed: bool,
    pub line_width: f64,
    pub label: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HeatmapData {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>, // row-major
    pub x_extent: Option<(f64, f64)>,
    pub y_extent: Option<(f64, f64)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuiverData {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub color: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceData {
    pub rows: usize,
    pub cols: usize,
    pub x: Vec<f64>, // row-major rows x cols
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    pub wireframe_only: bool, // false = surf, true = mesh
}

#[derive(Debug, Clone, PartialEq)]
pub struct Curve3D {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    pub color: String,
    pub dashed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlotPanel {
    pub title: String,
    pub xlabel: String,
    pub ylabel: String,
    pub zlabel: String,
    pub grid: bool,
    pub hold: bool,
    pub log_x: bool,
    pub log_y: bool,
    pub xlim: Option<(f64, f64)>,
    pub ylim: Option<(f64, f64)>,
    pub zlim: Option<(f64, f64)>,
    pub colormap: String,
    pub show_colorbar: bool,
    pub axis_equal: bool,
    pub axis_tight: bool,
    pub axis_off: bool,
    pub view_az: f64,
    pub view_el: f64,
    pub series: Vec<PlotSeries>,
    pub heatmap: Option<HeatmapData>,
    pub contour: Option<ContourData>,
    pub quiver: Option<QuiverData>,
    pub surface: Option<SurfaceData>,
    pub curves_3d: Vec<Curve3D>,
}

impl Default for PlotPanel {
    fn default() -> Self {
        Self {
            title: String::new(),
            xlabel: String::new(),
            ylabel: String::new(),
            zlabel: String::new(),
            grid: true,
            hold: false,
            log_x: false,
            log_y: false,
            xlim: None,
            ylim: None,
            zlim: None,
            colormap: "viridis".to_string(),
            show_colorbar: true,
            axis_equal: false,
            axis_tight: false,
            axis_off: false,
            view_az: -37.5,
            view_el: 30.0,
            series: Vec::new(),
            heatmap: None,
            contour: None,
            quiver: None,
            surface: None,
            curves_3d: Vec::new(),
        }
    }
}

impl PlotPanel {
    pub fn is_empty(&self) -> bool {
        self.series.is_empty()
            && self.heatmap.is_none()
            && self.contour.is_none()
            && self.quiver.is_none()
            && self.surface.is_none()
            && self.curves_3d.is_empty()
    }

    pub fn is_3d(&self) -> bool {
        self.surface.is_some() || !self.curves_3d.is_empty()
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn next_color(&self) -> String {
        PALETTE[self.series.len() % PALETTE.len()].to_string()
    }

    pub fn add_series(
        &mut self,
        kind: PlotKind,
        x: Vec<f64>,
        y: Vec<f64>,
        style: Option<&str>,
    ) {
        if !self.hold {
            self.series.clear();
            self.heatmap = None;
        }
        let (color, dashed) = parse_style(style, &self.next_color());
        self.series.push(PlotSeries {
            kind,
            x,
            y,
            err: Vec::new(),
            color,
            dashed,
            line_width: 1.8,
            label: None,
        });
    }

    pub fn add_errorbar(
        &mut self,
        x: Vec<f64>,
        y: Vec<f64>,
        err: Vec<f64>,
        style: Option<&str>,
    ) {
        if !self.hold {
            self.series.clear();
            self.heatmap = None;
        }
        let (color, dashed) = parse_style(style, &self.next_color());
        self.series.push(PlotSeries {
            kind: PlotKind::ErrorBar,
            x,
            y,
            err,
            color,
            dashed,
            line_width: 1.6,
            label: None,
        });
    }

    pub fn set_heatmap(&mut self, m: &Matrix) {
        self.set_heatmap_with_extents(m, None, None);
    }

    pub fn set_heatmap_with_extents(
        &mut self,
        m: &Matrix,
        x_extent: Option<(f64, f64)>,
        y_extent: Option<(f64, f64)>,
    ) {
        let mut row_major = Vec::with_capacity(m.len());
        for r in 0..m.rows() {
            for c in 0..m.cols() {
                row_major.push(m.get(r, c));
            }
        }
        self.series.clear();
        self.contour = None;
        self.heatmap = Some(HeatmapData {
            rows: m.rows(),
            cols: m.cols(),
            data: row_major,
            x_extent,
            y_extent,
        });
    }

    pub fn set_contour(&mut self, x: Vec<f64>, y: Vec<f64>, m: &Matrix, levels: usize) {
        let mut row_major = Vec::with_capacity(m.len());
        for r in 0..m.rows() {
            for c in 0..m.cols() {
                row_major.push(m.get(r, c));
            }
        }
        self.series.clear();
        self.heatmap = None;
        self.contour = Some(ContourData {
            x,
            y,
            z: row_major,
            levels: levels.clamp(2, 20),
        });
    }

    pub fn set_legend(&mut self, labels: Vec<String>) {
        for (ser, lbl) in self.series.iter_mut().zip(labels.into_iter()) {
            ser.label = Some(lbl);
        }
    }

    pub fn set_quiver(&mut self, x: Vec<f64>, y: Vec<f64>, u: Vec<f64>, v: Vec<f64>) {
        if !self.hold {
            self.series.clear();
            self.heatmap = None;
            self.contour = None;
            self.surface = None;
            self.curves_3d.clear();
        }
        self.quiver = Some(QuiverData {
            x,
            y,
            u,
            v,
            color: "#0072bd".to_string(),
        });
    }

    pub fn set_surface(
        &mut self,
        x_mat: &Matrix,
        y_mat: &Matrix,
        z_mat: &Matrix,
        wireframe_only: bool,
    ) {
        let rows = z_mat.rows();
        let cols = z_mat.cols();
        let mut x = Vec::with_capacity(rows * cols);
        let mut y = Vec::with_capacity(rows * cols);
        let mut z = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                x.push(x_mat.get(r, c));
                y.push(y_mat.get(r, c));
                z.push(z_mat.get(r, c));
            }
        }
        if !self.hold {
            self.series.clear();
            self.heatmap = None;
            self.contour = None;
            self.quiver = None;
            self.curves_3d.clear();
        }
        self.surface = Some(SurfaceData {
            rows,
            cols,
            x,
            y,
            z,
            wireframe_only,
        });
    }

    pub fn add_curve_3d(&mut self, x: Vec<f64>, y: Vec<f64>, z: Vec<f64>, style: Option<&str>) {
        if !self.hold {
            self.series.clear();
            self.heatmap = None;
            self.contour = None;
            self.quiver = None;
            self.surface = None;
            self.curves_3d.clear();
        }
        let default_col = PALETTE[self.curves_3d.len() % PALETTE.len()];
        let (color, dashed) = parse_style(style, default_col);
        self.curves_3d.push(Curve3D {
            x,
            y,
            z,
            color,
            dashed,
        });
    }
}

/// Multi-panel figure supporting `subplot(rows, cols, idx)` as well as single-panel plots.
#[derive(Debug, Clone, PartialEq)]
pub struct PlotFigure {
    pub grid_rows: usize,
    pub grid_cols: usize,
    pub active: usize,
    pub panels: Vec<PlotPanel>,
}

const PALETTE: &[&str] = &[
    "#0072bd", // classic MATLAB blue
    "#d95319", // burnt orange
    "#edb120", // ochre
    "#7e2f8e", // purple
    "#77ac30", // green
    "#4dbeee", // cyan
];

impl Default for PlotFigure {
    fn default() -> Self {
        Self {
            grid_rows: 1,
            grid_cols: 1,
            active: 0,
            panels: vec![PlotPanel::default()],
        }
    }
}

impl PlotFigure {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn subplot(&mut self, rows: usize, cols: usize, index_1based: usize) {
        let rows = rows.max(1);
        let cols = cols.max(1);
        let total = rows * cols;
        if self.grid_rows != rows || self.grid_cols != cols || self.panels.len() != total {
            self.grid_rows = rows;
            self.grid_cols = cols;
            self.panels = vec![PlotPanel::default(); total];
        }
        self.active = index_1based.saturating_sub(1).min(total - 1);
    }

    pub fn panel(&self) -> &PlotPanel {
        &self.panels[self.active]
    }

    pub fn panel_mut(&mut self) -> &mut PlotPanel {
        &mut self.panels[self.active]
    }

    pub fn is_empty(&self) -> bool {
        self.panels.iter().all(PlotPanel::is_empty)
    }

    pub fn add_series(
        &mut self,
        kind: PlotKind,
        x: Vec<f64>,
        y: Vec<f64>,
        style: Option<&str>,
    ) {
        self.panel_mut().add_series(kind, x, y, style);
    }

    /// Serialize figure metadata and panels to a JSON string (zero external dependencies).
    pub fn to_json(&self) -> String {
        let mut s = format!(
            "{{\"rows\":{},\"cols\":{},\"panels\":[",
            self.grid_rows, self.grid_cols
        );
        for (pi, panel) in self.panels.iter().enumerate() {
            if pi > 0 {
                s.push(',');
            }
            s.push('{');
            s.push_str(&format!("\"title\":{},", json_str(&panel.title)));
            s.push_str(&format!("\"xlabel\":{},", json_str(&panel.xlabel)));
            s.push_str(&format!("\"ylabel\":{},", json_str(&panel.ylabel)));
            s.push_str(&format!("\"grid\":{},", panel.grid));
            s.push_str(&format!("\"has_heatmap\":{},", panel.heatmap.is_some()));
            s.push_str("\"series\":[");
            for (i, ser) in panel.series.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push('{');
                s.push_str(&format!("\"kind\":{},", json_str(ser.kind.as_str())));
                s.push_str(&format!("\"color\":{},", json_str(&ser.color)));
                s.push_str(&format!("\"dashed\":{},", ser.dashed));
                if let Some(lbl) = &ser.label {
                    s.push_str(&format!("\"label\":{},", json_str(lbl)));
                }
                s.push_str("\"x\":");
                push_f64_array(&mut s, &ser.x);
                s.push_str(",\"y\":");
                push_f64_array(&mut s, &ser.y);
                s.push('}');
            }
            s.push_str("]}");
        }
        s.push_str("]}");
        s
    }

    /// Render this figure (all subplots) directly to a clean, white-background SVG string.
    pub fn to_svg(&self, width: f64, height: f64) -> String {
        if self.is_empty() {
            return String::new();
        }
        let total_h = height * (self.grid_rows as f64).min(2.2);
        let cell_w = width / self.grid_cols as f64;
        let cell_h = total_h / self.grid_rows as f64;

        let mut svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width} {total_h}\" width=\"100%\" height=\"100%\" style=\"background:#ffffff;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;\">"
        );

        for r in 0..self.grid_rows {
            for c in 0..self.grid_cols {
                let idx = r * self.grid_cols + c;
                if let Some(panel) = self.panels.get(idx) {
                    if !panel.is_empty() {
                        let ox = c as f64 * cell_w;
                        let oy = r as f64 * cell_h;
                        svg.push_str(&format!("<g transform=\"translate({ox:.1},{oy:.1})\">"));
                        render_panel(&mut svg, panel, cell_w, cell_h);
                        svg.push_str("</g>");
                    }
                }
            }
        }

        svg.push_str("</svg>");
        svg
    }
}

fn render_panel(svg: &mut String, panel: &PlotPanel, width: f64, height: f64) {
    if panel.is_3d() {
        render_panel_3d(svg, panel, width, height);
        return;
    }

    let pad_l = 58.0;
    let pad_r = if panel.heatmap.is_some() { 54.0 } else { 20.0 };
    let pad_t = if panel.title.is_empty() { 20.0 } else { 36.0 };
    let pad_b = if panel.xlabel.is_empty() { 36.0 } else { 48.0 };
    let pw = (width - pad_l - pad_r).max(40.0);
    let ph = (height - pad_t - pad_b).max(40.0);

    // Heatmap (`imagesc`) rendering path
    if let Some(hm) = &panel.heatmap {
        if hm.rows == 0 || hm.cols == 0 {
            return;
        }
        let mut vmin = f64::INFINITY;
        let mut vmax = f64::NEG_INFINITY;
        for &v in &hm.data {
            if v.is_finite() {
                vmin = vmin.min(v);
                vmax = vmax.max(v);
            }
        }
        if !vmin.is_finite() {
            return;
        }
        let span = (vmax - vmin).abs().max(1e-12);
        let cw = pw / hm.cols as f64;
        let ch = ph / hm.rows as f64;

        for r in 0..hm.rows {
            for c in 0..hm.cols {
                let v = hm.data[r * hm.cols + c];
                let t = ((v - vmin) / span).clamp(0.0, 1.0);
                let col = colormap_hex(&panel.colormap, t);
                svg.push_str(&format!(
                    "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\" fill=\"{col}\"/>",
                    pad_l + c as f64 * cw,
                    pad_t + r as f64 * ch,
                    cw + 0.4,
                    ch + 0.4
                ));
            }
        }
        svg.push_str(&format!(
            "<rect x=\"{pad_l}\" y=\"{pad_t}\" width=\"{pw}\" height=\"{ph}\" fill=\"none\" stroke=\"#333333\" stroke-width=\"1\"/>"
        ));
        // Axis extent tick labels on heatmap when provided
        let (x0, x1) = hm.x_extent.unwrap_or((1.0, hm.cols as f64));
        let (y0, y1) = hm.y_extent.unwrap_or((1.0, hm.rows as f64));
        if !panel.axis_off {
            for ti in 0..=4 {
                let frac = ti as f64 / 4.0;
                let tx = pad_l + frac * pw;
                let xv = x0 + frac * (x1 - x0);
                svg.push_str(&format!(
                    "<text x=\"{tx:.1}\" y=\"{:.1}\" text-anchor=\"middle\" fill=\"#4b5563\" font-size=\"10\">{}</text>",
                    pad_t + ph + 14.0,
                    fmt_tick(xv)
                ));
                let ty = pad_t + frac * ph;
                let yv = y0 + frac * (y1 - y0);
                svg.push_str(&format!(
                    "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" fill=\"#4b5563\" font-size=\"10\">{}</text>",
                    pad_l - 6.0,
                    ty + 3.5,
                    fmt_tick(yv)
                ));
            }
        }

        if panel.show_colorbar {
            let cb_x = pad_l + pw + 10.0;
            let cb_w = 10.0;
            let steps = 20usize;
            let sh = ph / steps as f64;
            for i in 0..steps {
                let t = 1.0 - (i as f64 / (steps - 1) as f64);
                let col = colormap_hex(&panel.colormap, t);
                svg.push_str(&format!(
                    "<rect x=\"{cb_x:.1}\" y=\"{:.2}\" width=\"{cb_w}\" height=\"{:.2}\" fill=\"{col}\"/>",
                    pad_t + i as f64 * sh,
                    sh + 0.5
                ));
            }
            svg.push_str(&format!(
                "<rect x=\"{cb_x:.1}\" y=\"{pad_t}\" width=\"{cb_w}\" height=\"{ph}\" fill=\"none\" stroke=\"#333333\" stroke-width=\"0.8\"/>"
            ));
            svg.push_str(&format!(
                "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"#4b5563\" font-size=\"10\">{}</text>",
                cb_x + cb_w + 4.0,
                pad_t + 8.0,
                fmt_tick(vmax)
            ));
            svg.push_str(&format!(
                "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"#4b5563\" font-size=\"10\">{}</text>",
                cb_x + cb_w + 4.0,
                pad_t + ph,
                fmt_tick(vmin)
            ));
        }
        render_labels(svg, panel, pad_l, pw, pad_t, ph, height);
        return;
    }

    // Contour (`contour`) rendering path via Marching Squares
    if let Some(ct) = &panel.contour {
        let rows = ct.y.len();
        let cols = ct.x.len();
        if rows < 2 || cols < 2 {
            return;
        }
        let xmin = ct.x.iter().copied().fold(f64::INFINITY, f64::min);
        let xmax = ct.x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let ymin = ct.y.iter().copied().fold(f64::INFINITY, f64::min);
        let ymax = ct.y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let zmin = ct.z.iter().copied().fold(f64::INFINITY, f64::min);
        let zmax = ct.z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if (xmax - xmin).abs() < 1e-12 || (ymax - ymin).abs() < 1e-12 || (zmax - zmin).abs() < 1e-12 {
            return;
        }
        let sx = |x: f64| pad_l + ((x - xmin) / (xmax - xmin)) * pw;
        let sy = |y: f64| pad_t + ph - ((y - ymin) / (ymax - ymin)) * ph;

        svg.push_str(&format!(
            "<rect x=\"{pad_l}\" y=\"{pad_t}\" width=\"{pw}\" height=\"{ph}\" fill=\"#ffffff\" stroke=\"#333333\" stroke-width=\"1\"/>"
        ));
        for li in 1..=ct.levels {
            let frac = li as f64 / (ct.levels as f64 + 1.0);
            let lvl = zmin + frac * (zmax - zmin);
            let color = viridis_hex(frac);
            for r in 0..(rows - 1) {
                for c in 0..(cols - 1) {
                    let v00 = ct.z[r * cols + c];
                    let v10 = ct.z[r * cols + (c + 1)];
                    let v11 = ct.z[(r + 1) * cols + (c + 1)];
                    let v01 = ct.z[(r + 1) * cols + c];
                    let x0 = ct.x[c];
                    let x1 = ct.x[c + 1];
                    let y0 = ct.y[r];
                    let y1 = ct.y[r + 1];
                    let mut pts = Vec::with_capacity(4);
                    let interp = |a: f64, b: f64, va: f64, vb: f64| -> f64 {
                        a + ((lvl - va) / (vb - va)) * (b - a)
                    };
                    if (v00 < lvl) != (v10 < lvl) {
                        pts.push((interp(x0, x1, v00, v10), y0));
                    }
                    if (v10 < lvl) != (v11 < lvl) {
                        pts.push((x1, interp(y0, y1, v10, v11)));
                    }
                    if (v11 < lvl) != (v01 < lvl) {
                        pts.push((interp(x1, x0, v11, v01), y1));
                    }
                    if (v01 < lvl) != (v00 < lvl) {
                        pts.push((x0, interp(y1, y0, v01, v00)));
                    }
                    if pts.len() >= 2 {
                        svg.push_str(&format!(
                            "<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"{color}\" stroke-width=\"1.4\"/>",
                            sx(pts[0].0),
                            sy(pts[0].1),
                            sx(pts[1].0),
                            sy(pts[1].1)
                        ));
                    }
                }
            }
        }
        render_labels(svg, panel, pad_l, pw, pad_t, ph, height);
        return;
    }

    // Standard XY line/scatter/bar rendering path
    let tx = |x: f64| if panel.log_x { x.max(1e-30).log10() } else { x };
    let ty = |y: f64| if panel.log_y { y.max(1e-30).log10() } else { y };

    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    for ser in &panel.series {
        for &x in &ser.x {
            let xv = tx(x);
            if xv.is_finite() {
                xmin = xmin.min(xv);
                xmax = xmax.max(xv);
            }
        }
        for (i, &y) in ser.y.iter().enumerate() {
            let yv = ty(y);
            if yv.is_finite() {
                ymin = ymin.min(yv);
                ymax = ymax.max(yv);
            }
            if ser.kind == PlotKind::ErrorBar {
                if let Some(&e) = ser.err.get(i) {
                    let e = e.abs();
                    if e.is_finite() {
                        let ylo = ty(y - e);
                        let yhi = ty(y + e);
                        if ylo.is_finite() {
                            ymin = ymin.min(ylo);
                        }
                        if yhi.is_finite() {
                            ymax = ymax.max(yhi);
                        }
                    }
                }
            }
        }
        if ser.kind == PlotKind::Bar && !panel.log_y {
            ymin = ymin.min(0.0);
        }
    }
    if let Some(q) = &panel.quiver {
        for &x in &q.x {
            let xv = tx(x);
            if xv.is_finite() {
                xmin = xmin.min(xv);
                xmax = xmax.max(xv);
            }
        }
        for &y in &q.y {
            let yv = ty(y);
            if yv.is_finite() {
                ymin = ymin.min(yv);
                ymax = ymax.max(yv);
            }
        }
        if xmin.is_finite() && xmax.is_finite() {
            let mx = (xmax - xmin).abs().max(1.0) * 0.08;
            xmin -= mx;
            xmax += mx;
        }
        if ymin.is_finite() && ymax.is_finite() {
            let my = (ymax - ymin).abs().max(1.0) * 0.08;
            ymin -= my;
            ymax += my;
        }
    }
    if let Some((lo, hi)) = panel.xlim {
        xmin = tx(lo);
        xmax = tx(hi);
    }
    if let Some((lo, hi)) = panel.ylim {
        ymin = ty(lo);
        ymax = ty(hi);
    } else if !panel.axis_tight && ymin.is_finite() && ymax.is_finite() && (ymax - ymin).abs() > 1e-12 {
        let margin = (ymax - ymin) * 0.06;
        ymin -= margin;
        ymax += margin;
    }

    if !xmin.is_finite() || !ymin.is_finite() {
        return;
    }
    if (xmax - xmin).abs() < 1e-12 {
        xmin -= 1.0;
        xmax += 1.0;
    }
    if (ymax - ymin).abs() < 1e-12 {
        ymin -= 1.0;
        ymax += 1.0;
    }
    if panel.axis_equal && pw > 0.0 && ph > 0.0 {
        let rx = (xmax - xmin) / pw;
        let ry = (ymax - ymin) / ph;
        if rx > ry {
            let cy = 0.5 * (ymin + ymax);
            let half = 0.5 * rx * ph;
            ymin = cy - half;
            ymax = cy + half;
        } else if ry > rx {
            let cx = 0.5 * (xmin + xmax);
            let half = 0.5 * ry * pw;
            xmin = cx - half;
            xmax = cx + half;
        }
    }

    let sx = |x: f64| pad_l + ((tx(x) - xmin) / (xmax - xmin)) * pw;
    let sy = |y: f64| pad_t + ph - ((ty(y) - ymin) / (ymax - ymin)) * ph;

    // Frame, grid & ticks (hidden when axis_off is true)
    if !panel.axis_off {
        svg.push_str(&format!(
            "<rect x=\"{pad_l}\" y=\"{pad_t}\" width=\"{pw}\" height=\"{ph}\" fill=\"#ffffff\" stroke=\"#333333\" stroke-width=\"1\"/>"
        ));

        let ticks = 5usize;
        for i in 0..=ticks {
            let frac = i as f64 / ticks as f64;
            let xv_raw = xmin + frac * (xmax - xmin);
            let yv_raw = ymin + frac * (ymax - ymin);
            let xv = if panel.log_x { 10f64.powf(xv_raw) } else { xv_raw };
            let yv = if panel.log_y { 10f64.powf(yv_raw) } else { yv_raw };
            let px = pad_l + frac * pw;
            let py = pad_t + ph - frac * ph;

            if panel.grid && i > 0 && i < ticks {
                svg.push_str(&format!(
                    "<line x1=\"{px:.1}\" y1=\"{pad_t}\" x2=\"{px:.1}\" y2=\"{:.1}\" stroke=\"#e5e7eb\" stroke-width=\"1\"/>",
                    pad_t + ph
                ));
                svg.push_str(&format!(
                    "<line x1=\"{pad_l}\" y1=\"{py:.1}\" x2=\"{:.1}\" y2=\"{py:.1}\" stroke=\"#e5e7eb\" stroke-width=\"1\"/>",
                    pad_l + pw
                ));
            }
            svg.push_str(&format!(
                "<text x=\"{px:.1}\" y=\"{:.1}\" fill=\"#4b5563\" font-size=\"10.5\" text-anchor=\"middle\">{}</text>",
                pad_t + ph + 15.0,
                fmt_tick(xv)
            ));
            svg.push_str(&format!(
                "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"#4b5563\" font-size=\"10.5\" text-anchor=\"end\">{}</text>",
                pad_l - 7.0,
                py + 3.5,
                fmt_tick(yv)
            ));
        }
    }

    // Series
    for ser in &panel.series {
        let n = ser.x.len().min(ser.y.len());
        match ser.kind {
            PlotKind::Line => {
                let mut pts = String::new();
                for i in 0..n {
                    if ser.x[i].is_finite() && ser.y[i].is_finite() {
                        let px = sx(ser.x[i]).clamp(pad_l, pad_l + pw);
                        let py = sy(ser.y[i]).clamp(pad_t, pad_t + ph);
                        pts.push_str(&format!("{px:.2},{py:.2} "));
                    }
                }
                let dash = if ser.dashed {
                    " stroke-dasharray=\"6,4\""
                } else {
                    ""
                };
                svg.push_str(&format!(
                    "<polyline fill=\"none\" stroke=\"{}\" stroke-width=\"{:.2}\" stroke-linejoin=\"round\" stroke-linecap=\"round\"{dash} points=\"{pts}\"/>",
                    ser.color, ser.line_width
                ));
            }
            PlotKind::Scatter => {
                for i in 0..n {
                    if ser.x[i].is_finite() && ser.y[i].is_finite() {
                        svg.push_str(&format!(
                            "<circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"3.2\" fill=\"{}\" fill-opacity=\"0.85\"/>",
                            sx(ser.x[i]),
                            sy(ser.y[i]),
                            ser.color
                        ));
                    }
                }
            }
            PlotKind::Bar => {
                let bw = (pw / (n.max(1) as f64) * 0.65).clamp(4.0, 42.0);
                let y0 = sy(0.0).clamp(pad_t, pad_t + ph);
                for i in 0..n {
                    if ser.x[i].is_finite() && ser.y[i].is_finite() {
                        let cx = sx(ser.x[i]);
                        let cy = sy(ser.y[i]).clamp(pad_t, pad_t + ph);
                        let top = cy.min(y0);
                        let h = (cy - y0).abs().max(1.0);
                        svg.push_str(&format!(
                            "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"{bw:.2}\" height=\"{h:.2}\" fill=\"{}\" fill-opacity=\"0.85\"/>",
                            cx - bw * 0.5,
                            top,
                            ser.color
                        ));
                    }
                }
            }
            PlotKind::Stem => {
                let y0 = sy(0.0).clamp(pad_t, pad_t + ph);
                svg.push_str(&format!(
                    "<line x1=\"{pad_l}\" y1=\"{y0:.2}\" x2=\"{:.2}\" y2=\"{y0:.2}\" stroke=\"#9ca3af\" stroke-width=\"1\"/>",
                    pad_l + pw
                ));
                for i in 0..n {
                    if ser.x[i].is_finite() && ser.y[i].is_finite() {
                        let cx = sx(ser.x[i]);
                        let cy = sy(ser.y[i]).clamp(pad_t, pad_t + ph);
                        svg.push_str(&format!(
                            "<line x1=\"{cx:.2}\" y1=\"{y0:.2}\" x2=\"{cx:.2}\" y2=\"{cy:.2}\" stroke=\"{}\" stroke-width=\"1.4\"/>",
                            ser.color
                        ));
                        svg.push_str(&format!(
                            "<circle cx=\"{cx:.2}\" cy=\"{cy:.2}\" r=\"3.2\" fill=\"#ffffff\" stroke=\"{}\" stroke-width=\"1.6\"/>",
                            ser.color
                        ));
                    }
                }
            }
            PlotKind::ErrorBar => {
                let mut pts = String::new();
                for i in 0..n {
                    if ser.x[i].is_finite() && ser.y[i].is_finite() {
                        let cx = sx(ser.x[i]);
                        let cy = sy(ser.y[i]).clamp(pad_t, pad_t + ph);
                        if !pts.is_empty() {
                            pts.push(' ');
                        }
                        pts.push_str(&format!("{cx:.2},{cy:.2}"));
                        let e = ser.err.get(i).copied().unwrap_or(0.0).abs();
                        if e > 0.0 && e.is_finite() {
                            let y_hi = sy(ser.y[i] + e).clamp(pad_t, pad_t + ph);
                            let y_lo = sy(ser.y[i] - e).clamp(pad_t, pad_t + ph);
                            svg.push_str(&format!(
                                "<line x1=\"{cx:.2}\" y1=\"{y_lo:.2}\" x2=\"{cx:.2}\" y2=\"{y_hi:.2}\" stroke=\"{}\" stroke-width=\"1.2\"/>",
                                ser.color
                            ));
                            svg.push_str(&format!(
                                "<line x1=\"{:.2}\" y1=\"{y_lo:.2}\" x2=\"{:.2}\" y2=\"{y_lo:.2}\" stroke=\"{}\" stroke-width=\"1.2\"/>",
                                cx - 3.5, cx + 3.5, ser.color
                            ));
                            svg.push_str(&format!(
                                "<line x1=\"{:.2}\" y1=\"{y_hi:.2}\" x2=\"{:.2}\" y2=\"{y_hi:.2}\" stroke=\"{}\" stroke-width=\"1.2\"/>",
                                cx - 3.5, cx + 3.5, ser.color
                            ));
                        }
                    }
                }
                if !pts.is_empty() {
                    svg.push_str(&format!(
                        "<polyline fill=\"none\" stroke=\"{}\" stroke-width=\"{:.2}\" points=\"{pts}\"/>",
                        ser.color, ser.line_width
                    ));
                }
            }
            PlotKind::Fill => {
                let mut pts = String::new();
                for i in 0..n {
                    if ser.x[i].is_finite() && ser.y[i].is_finite() {
                        if !pts.is_empty() {
                            pts.push(' ');
                        }
                        pts.push_str(&format!(
                            "{:.2},{:.2}",
                            sx(ser.x[i]),
                            sy(ser.y[i]).clamp(pad_t, pad_t + ph)
                        ));
                    }
                }
                if !pts.is_empty() {
                    svg.push_str(&format!(
                        "<polygon points=\"{pts}\" fill=\"{}\" fill-opacity=\"0.38\" stroke=\"{}\" stroke-width=\"{:.2}\"/>",
                        ser.color, ser.color, ser.line_width
                    ));
                }
            }
        }
    }
    if let Some(q) = &panel.quiver {
        let n = q.x.len().min(q.y.len()).min(q.u.len()).min(q.v.len());
        let max_mag = (0..n)
            .map(|i| q.u[i].hypot(q.v[i]))
            .filter(|m| m.is_finite())
            .fold(0.0f64, f64::max);
        let arrow_px = (pw.min(ph) / (n as f64).sqrt().max(2.0)) * 0.72;
        if max_mag > 1e-12 {
            for i in 0..n {
                if !q.x[i].is_finite() || !q.y[i].is_finite() {
                    continue;
                }
                let x0 = sx(q.x[i]);
                let y0 = sy(q.y[i]);
                let dx = (q.u[i] / max_mag) * arrow_px;
                let dy = -(q.v[i] / max_mag) * arrow_px;
                let x1 = x0 + dx;
                let y1 = y0 + dy;
                let len = dx.hypot(dy);
                if len < 1.0 {
                    continue;
                }
                svg.push_str(&format!(
                    "<line x1=\"{x0:.2}\" y1=\"{y0:.2}\" x2=\"{x1:.2}\" y2=\"{y1:.2}\" stroke=\"{}\" stroke-width=\"1.2\"/>",
                    q.color
                ));
                let ux = dx / len;
                let uy = dy / len;
                let head = (len * 0.28).clamp(3.0, 6.5);
                let hw = head * 0.55;
                let ax = x1 - ux * head + uy * hw;
                let ay = y1 - uy * head - ux * hw;
                let bx = x1 - ux * head - uy * hw;
                let by = y1 - uy * head + ux * hw;
                svg.push_str(&format!(
                    "<polygon points=\"{x1:.2},{y1:.2} {ax:.2},{ay:.2} {bx:.2},{by:.2}\" fill=\"{}\"/>",
                    q.color
                ));
            }
        }
    }

    // Legend (if any series has a label)
    let labeled: Vec<&PlotSeries> = panel
        .series
        .iter()
        .filter(|s| s.label.is_some())
        .collect();
    if !labeled.is_empty() {
        let max_chars = labeled
            .iter()
            .map(|s| s.label.as_deref().unwrap_or("").len())
            .max()
            .unwrap_or(4);
        let lw = (max_chars as f64 * 6.5 + 34.0).min(pw * 0.48);
        let lh = labeled.len() as f64 * 16.0 + 10.0;
        let lx = pad_l + pw - lw - 8.0;
        let ly = pad_t + 8.0;
        svg.push_str(&format!(
            "<rect x=\"{lx:.1}\" y=\"{ly:.1}\" width=\"{lw:.1}\" height=\"{lh:.1}\" fill=\"#ffffff\" fill-opacity=\"0.92\" stroke=\"#d1d5db\" stroke-width=\"1\"/>"
        ));
        for (i, ser) in labeled.iter().enumerate() {
            let cy = ly + 13.0 + i as f64 * 16.0;
            let dash = if ser.dashed {
                " stroke-dasharray=\"4,2\""
            } else {
                ""
            };
            svg.push_str(&format!(
                "<line x1=\"{:.1}\" y1=\"{cy:.1}\" x2=\"{:.1}\" y2=\"{cy:.1}\" stroke=\"{}\" stroke-width=\"2\"{dash}/>",
                lx + 8.0,
                lx + 24.0,
                ser.color
            ));
            svg.push_str(&format!(
                "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"#1f2937\" font-size=\"10.5\">{}</text>",
                lx + 30.0,
                cy + 3.5,
                xml_escape(ser.label.as_deref().unwrap_or(""))
            ));
        }
    }

    render_labels(svg, panel, pad_l, pw, pad_t, ph, height);
}

/// Pure-Rust 3D perspective/orthographic SVG renderer for `surf`, `mesh`, and `plot3`.
fn render_panel_3d(svg: &mut String, panel: &PlotPanel, width: f64, height: f64) {
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    let mut zmin = f64::INFINITY;
    let mut zmax = f64::NEG_INFINITY;

    let mut absorb = |x: f64, y: f64, z: f64| {
        if x.is_finite() && y.is_finite() && z.is_finite() {
            xmin = xmin.min(x);
            xmax = xmax.max(x);
            ymin = ymin.min(y);
            ymax = ymax.max(y);
            zmin = zmin.min(z);
            zmax = zmax.max(z);
        }
    };

    if let Some(s) = &panel.surface {
        for i in 0..(s.rows * s.cols) {
            absorb(s.x[i], s.y[i], s.z[i]);
        }
    }
    for c in &panel.curves_3d {
        let n = c.x.len().min(c.y.len()).min(c.z.len());
        for i in 0..n {
            absorb(c.x[i], c.y[i], c.z[i]);
        }
    }
    if !xmin.is_finite() || !ymin.is_finite() || !zmin.is_finite() {
        return;
    }
    if (xmax - xmin).abs() < 1e-12 {
        xmin -= 1.0;
        xmax += 1.0;
    }
    if (ymax - ymin).abs() < 1e-12 {
        ymin -= 1.0;
        ymax += 1.0;
    }
    if (zmax - zmin).abs() < 1e-12 {
        zmin -= 1.0;
        zmax += 1.0;
    }

    let az = panel.view_az.to_radians();
    let el = panel.view_el.to_radians();
    let (saz, caz) = az.sin_cos();
    let (sel, cel) = el.sin_cos();

    let pad_t = if panel.title.is_empty() { 18.0 } else { 34.0 };
    let cx = width * 0.5;
    let cy = pad_t + (height - pad_t - 24.0) * 0.50;
    let scale = (width.min(height - pad_t) * 0.28).max(26.0);

    // Project world (x, y, z) -> (screen_x, screen_y, cam_depth) where larger depth = closer
    let project = |x: f64, y: f64, z: f64| -> (f64, f64, f64) {
        let nx = 2.0 * (x - xmin) / (xmax - xmin) - 1.0;
        let ny = 2.0 * (y - ymin) / (ymax - ymin) - 1.0;
        let nz = 2.0 * (z - zmin) / (zmax - zmin) - 1.0;
        // Standard MATLAB camera basis: az rotates around Z, el tilts up from XY plane
        let rx = nx * caz - ny * saz;
        let ry = nx * saz + ny * caz;
        let sx = rx;
        let sy = ry * sel + nz * cel;
        let depth = -ry * cel + nz * sel;
        let persp = 1.0 / (1.0 - 0.12 * depth).max(0.4);
        (cx + sx * scale * persp, cy - sy * scale * persp, depth)
    };

    // Draw 3D bounding-box wireframe back edges & floor grid
    let corners = [
        (xmin, ymin, zmin),
        (xmax, ymin, zmin),
        (xmax, ymax, zmin),
        (xmin, ymax, zmin),
        (xmin, ymin, zmax),
        (xmax, ymin, zmax),
        (xmax, ymax, zmax),
        (xmin, ymax, zmax),
    ];
    let pc: Vec<(f64, f64, f64)> = corners.iter().map(|&(x, y, z)| project(x, y, z)).collect();
    let edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
    ];
    if panel.grid {
        for k in 1..4 {
            let f = k as f64 / 4.0;
            let xg = xmin + f * (xmax - xmin);
            let yg = ymin + f * (ymax - ymin);
            let (ax, ay, _) = project(xg, ymin, zmin);
            let (bx, by, _) = project(xg, ymax, zmin);
            svg.push_str(&format!(
                "<line x1=\"{ax:.1}\" y1=\"{ay:.1}\" x2=\"{bx:.1}\" y2=\"{by:.1}\" stroke=\"#e5e7eb\" stroke-width=\"0.9\"/>"
            ));
            let (cx0, cy0, _) = project(xmin, yg, zmin);
            let (dx0, dy0, _) = project(xmax, yg, zmin);
            svg.push_str(&format!(
                "<line x1=\"{cx0:.1}\" y1=\"{cy0:.1}\" x2=\"{dx0:.1}\" y2=\"{dy0:.1}\" stroke=\"#e5e7eb\" stroke-width=\"0.9\"/>"
            ));
        }
    }
    for (a, b) in edges {
        let stroke = if a < 4 && b < 4 { "#9ca3af" } else { "#d1d5db" };
        svg.push_str(&format!(
            "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{stroke}\" stroke-width=\"0.9\"/>",
            pc[a].0, pc[a].1, pc[b].0, pc[b].1
        ));
    }

    // Painter's algorithm for `surf` / `mesh`
    if let Some(s) = &panel.surface {
        if s.rows >= 2 && s.cols >= 2 {
            struct Quad {
                depth: f64,
                pts: [(f64, f64); 4],
                z_norm: f64,
            }
            let mut quads = Vec::with_capacity((s.rows - 1) * (s.cols - 1));
            let zspan = (zmax - zmin).abs().max(1e-12);
            for r in 0..(s.rows - 1) {
                for c in 0..(s.cols - 1) {
                    let i00 = r * s.cols + c;
                    let i10 = (r + 1) * s.cols + c;
                    let i11 = (r + 1) * s.cols + (c + 1);
                    let i01 = r * s.cols + (c + 1);
                    let zs = [s.z[i00], s.z[i10], s.z[i11], s.z[i01]];
                    if !zs.iter().all(|v| v.is_finite()) {
                        continue;
                    }
                    let p00 = project(s.x[i00], s.y[i00], s.z[i00]);
                    let p10 = project(s.x[i10], s.y[i10], s.z[i10]);
                    let p11 = project(s.x[i11], s.y[i11], s.z[i11]);
                    let p01 = project(s.x[i01], s.y[i01], s.z[i01]);
                    let depth = 0.25 * (p00.2 + p10.2 + p11.2 + p01.2);
                    let z_avg = 0.25 * (zs[0] + zs[1] + zs[2] + zs[3]);
                    quads.push(Quad {
                        depth,
                        pts: [(p00.0, p00.1), (p10.0, p10.1), (p11.0, p11.1), (p01.0, p01.1)],
                        z_norm: ((z_avg - zmin) / zspan).clamp(0.0, 1.0),
                    });
                }
            }
            // Sort far-to-near (ascending depth)
            quads.sort_by(|a, b| a.depth.total_cmp(&b.depth));
            for q in quads {
                let col = viridis_hex(q.z_norm);
                let (fill, stroke, sw) = if s.wireframe_only {
                    ("#ffffff", col.as_str(), "1.0")
                } else {
                    (col.as_str(), "#1f2937", "0.45")
                };
                let fill_op = if s.wireframe_only {
                    " fill-opacity=\"0.85\""
                } else {
                    ""
                };
                svg.push_str(&format!(
                    "<polygon points=\"{:.1},{:.1} {:.1},{:.1} {:.1},{:.1} {:.1},{:.1}\" fill=\"{fill}\"{fill_op} stroke=\"{stroke}\" stroke-width=\"{sw}\" stroke-linejoin=\"round\"/>",
                    q.pts[0].0, q.pts[0].1, q.pts[1].0, q.pts[1].1, q.pts[2].0, q.pts[2].1, q.pts[3].0, q.pts[3].1
                ));
            }
        }
    }

    // 3D curves (`plot3`)
    for curve in &panel.curves_3d {
        let n = curve.x.len().min(curve.y.len()).min(curve.z.len());
        let mut pts = String::new();
        for i in 0..n {
            if curve.x[i].is_finite() && curve.y[i].is_finite() && curve.z[i].is_finite() {
                let (px, py, _) = project(curve.x[i], curve.y[i], curve.z[i]);
                pts.push_str(&format!("{px:.2},{py:.2} "));
            }
        }
        let dash = if curve.dashed {
            " stroke-dasharray=\"6,4\""
        } else {
            ""
        };
        svg.push_str(&format!(
            "<polyline fill=\"none\" stroke=\"{}\" stroke-width=\"1.8\" stroke-linejoin=\"round\" stroke-linecap=\"round\"{dash} points=\"{pts}\"/>",
            curve.color
        ));
    }

    // Axis endpoint ticks & labels (clamped inside panel bounds so narrow subplots never clip)
    let xl = if panel.xlabel.is_empty() { "X" } else { &panel.xlabel };
    let yl = if panel.ylabel.is_empty() { "Y" } else { &panel.ylabel };
    let zl = if panel.zlabel.is_empty() { "Z" } else { &panel.zlabel };
    let (mx, my, _) = project(0.5 * (xmin + xmax), ymin, zmin);
    let (yx, yy, _) = project(xmax, 0.5 * (ymin + ymax), zmin);
    let (zx, zy, _) = project(xmin, ymin, 0.5 * (zmin + zmax));
    let x_lbl_x = mx.clamp(56.0, (width - 56.0).max(56.0));
    let x_lbl_y = (my + 15.0).min(height - 6.0);
    let y_lbl_x = (yx + 6.0).min(width - 6.0);
    let y_lbl_anchor = if y_lbl_x > width * 0.72 { "end" } else { "start" };
    let z_lbl_x = (zx - 6.0).max(6.0);
    let z_lbl_anchor = if z_lbl_x < width * 0.28 { "start" } else { "end" };
    svg.push_str(&format!(
        "<text x=\"{x_lbl_x:.1}\" y=\"{x_lbl_y:.1}\" fill=\"#374151\" font-size=\"10\" text-anchor=\"middle\">{} [{}, {}]</text>",
        xml_escape(xl),
        fmt_tick(xmin),
        fmt_tick(xmax)
    ));
    svg.push_str(&format!(
        "<text x=\"{y_lbl_x:.1}\" y=\"{:.1}\" fill=\"#374151\" font-size=\"10\" text-anchor=\"{y_lbl_anchor}\">{} [{}, {}]</text>",
        (yy + 11.0).min(height - 6.0),
        xml_escape(yl),
        fmt_tick(ymin),
        fmt_tick(ymax)
    ));
    svg.push_str(&format!(
        "<text x=\"{z_lbl_x:.1}\" y=\"{:.1}\" fill=\"#374151\" font-size=\"10\" text-anchor=\"{z_lbl_anchor}\">{} [{}, {}]</text>",
        zy.clamp(pad_t + 10.0, height - 10.0),
        xml_escape(zl),
        fmt_tick(zmin),
        fmt_tick(zmax)
    ));

    if !panel.title.is_empty() {
        svg.push_str(&format!(
            "<text x=\"{:.1}\" y=\"22\" fill=\"#111827\" font-size=\"12.5\" font-weight=\"600\" text-anchor=\"middle\">{}</text>",
            width * 0.5,
            xml_escape(&panel.title)
        ));
    }
}

fn render_labels(
    svg: &mut String,
    panel: &PlotPanel,
    pad_l: f64,
    pw: f64,
    pad_t: f64,
    ph: f64,
    height: f64,
) {
    if !panel.title.is_empty() {
        svg.push_str(&format!(
            "<text x=\"{:.1}\" y=\"22\" fill=\"#111827\" font-size=\"12.5\" font-weight=\"600\" text-anchor=\"middle\">{}</text>",
            pad_l + pw * 0.5,
            xml_escape(&panel.title)
        ));
    }
    if !panel.xlabel.is_empty() {
        svg.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"#374151\" font-size=\"11\" text-anchor=\"middle\">{}</text>",
            pad_l + pw * 0.5,
            height - 8.0,
            xml_escape(&panel.xlabel)
        ));
    }
    if !panel.ylabel.is_empty() {
        let cy = pad_t + ph * 0.5;
        svg.push_str(&format!(
            "<text x=\"15\" y=\"{cy:.1}\" fill=\"#374151\" font-size=\"11\" text-anchor=\"middle\" transform=\"rotate(-90 15 {cy:.1})\">{}</text>",
            xml_escape(&panel.ylabel)
        ));
    }
}

/// Compact Viridis-inspired 5-stop perceptual colormap.
fn viridis_hex(t: f64) -> String {
    const STOPS: [(u8, u8, u8); 5] = [
        (68, 1, 84),    // 0.00 deep purple
        (59, 82, 139),  // 0.25 slate blue
        (33, 145, 140), // 0.50 teal
        (94, 201, 98),  // 0.75 green
        (253, 231, 37), // 1.00 yellow
    ];
    let scaled = t.clamp(0.0, 1.0) * (STOPS.len() - 1) as f64;
    let idx = (scaled.floor() as usize).min(STOPS.len() - 2);
    let frac = scaled - idx as f64;
    let (r0, g0, b0) = STOPS[idx];
    let (r1, g1, b1) = STOPS[idx + 1];
    let r = (r0 as f64 + (r1 as f64 - r0 as f64) * frac).round() as u8;
    let g = (g0 as f64 + (g1 as f64 - g0 as f64) * frac).round() as u8;
    let b = (b0 as f64 + (b1 as f64 - b0 as f64) * frac).round() as u8;
    format!("#{r:02x}{g:02x}{b:02x}")
}

fn colormap_hex(name: &str, t: f64) -> String {
    let t = t.clamp(0.0, 1.0);
    let interp = |stops: &[(u8, u8, u8)]| -> String {
        let scaled = t * (stops.len() - 1) as f64;
        let idx = (scaled.floor() as usize).min(stops.len() - 2);
        let frac = scaled - idx as f64;
        let (r0, g0, b0) = stops[idx];
        let (r1, g1, b1) = stops[idx + 1];
        let r = (r0 as f64 + (r1 as f64 - r0 as f64) * frac).round() as u8;
        let g = (g0 as f64 + (g1 as f64 - g0 as f64) * frac).round() as u8;
        let b = (b0 as f64 + (b1 as f64 - b0 as f64) * frac).round() as u8;
        format!("#{r:02x}{g:02x}{b:02x}")
    };
    match name.to_ascii_lowercase().as_str() {
        "gray" | "grey" => {
            let v = (t * 255.0).round() as u8;
            format!("#{v:02x}{v:02x}{v:02x}")
        }
        "hot" => interp(&[(11, 0, 0), (190, 0, 0), (255, 110, 0), (255, 235, 0), (255, 255, 255)]),
        "cool" => interp(&[(0, 255, 255), (128, 128, 255), (255, 0, 255)]),
        "jet" | "turbo" => interp(&[
            (0, 0, 143),
            (0, 128, 255),
            (0, 255, 128),
            (255, 255, 0),
            (255, 96, 0),
            (128, 0, 0),
        ]),
        "parula" => interp(&[
            (62, 38, 168),
            (46, 135, 247),
            (31, 188, 195),
            (196, 193, 47),
            (249, 251, 20),
        ]),
        "bone" => interp(&[(0, 0, 0), (84, 84, 116), (167, 199, 199), (255, 255, 255)]),
        _ => viridis_hex(t),
    }
}

fn parse_style(style: Option<&str>, default_color: &str) -> (String, bool) {
    let Some(s) = style else {
        return (default_color.to_string(), false);
    };
    let dashed = s.contains("--") || s.contains(':');
    let color = if s.contains('r') {
        "#d62728"
    } else if s.contains('g') {
        "#2ca02c"
    } else if s.contains('b') {
        "#0072bd"
    } else if s.contains('y') {
        "#d97706"
    } else if s.contains('m') {
        "#9333ea"
    } else if s.contains('c') {
        "#0891b2"
    } else if s.contains('w') {
        "#9ca3af"
    } else if s.contains('k') {
        "#111827"
    } else {
        default_color
    };
    (color.to_string(), dashed)
}

fn fmt_tick(v: f64) -> String {
    if v.abs() < 1e-12 {
        "0".to_string()
    } else if v.abs() >= 1000.0 || v.abs() < 0.01 {
        format!("{v:.1e}")
    } else if v.fract().abs() < 1e-9 {
        format!("{}", v.round() as i64)
    } else {
        format!("{v:.2}")
    }
}

pub fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn push_f64_array(out: &mut String, vals: &[f64]) {
    out.push('[');
    for (i, &v) in vals.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        if v.is_finite() {
            out.push_str(&format!("{v}"));
        } else {
            out.push_str("null");
        }
    }
    out.push(']');
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

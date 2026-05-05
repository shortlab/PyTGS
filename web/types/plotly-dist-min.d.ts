// Stub module declaration for the pre-bundled Plotly distribution.
// The package is a UMD bundle with no types; we borrow them from `plotly.js`.
declare module "plotly.js-dist-min" {
  import Plotly from "plotly.js";
  export default Plotly;
}

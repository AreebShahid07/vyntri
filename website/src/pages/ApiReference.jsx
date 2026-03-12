import CodeBlock from '../components/CodeBlock'

export default function ApiReference() {
  return (
    <div className="doc-page">
      <h1>API Reference</h1>
      <p className="lead">Complete reference for all Vyntri modules, classes, and functions.</p>

      <nav className="doc-toc">
        <h3>Modules</h3>
        <ul>
          <li><a href="#pipeline">vyntri.core.pipeline</a></li>
          <li><a href="#config">vyntri.core.config</a></li>
          <li><a href="#io">vyntri.core.io</a></li>
          <li><a href="#engine">vyntri.dataset.engine</a></li>
          <li><a href="#pvi">vyntri.dataset.pvi</a></li>
          <li><a href="#backbones">vyntri.backbones.loader</a></li>
          <li><a href="#solvers">vyntri.solvers</a></li>
        </ul>
      </nav>

      {/* Pipeline */}
      <section id="pipeline">
        <h2>vyntri.core.pipeline</h2>
        <div className="api-class">
          <h3><code>VyntriPipeline(config=None)</code></h3>
          <p>End-to-end pipeline: Analyze → Select backbone → Extract features → Adapt solver → Predict.</p>

          <div className="api-method">
            <h4><code>run(dataset_path, solver_type="anacp", **solver_kwargs)</code></h4>
            <p>Run the full pipeline on a folder-per-class dataset.</p>
            <table className="param-table">
              <thead><tr><th>Parameter</th><th>Type</th><th>Description</th></tr></thead>
              <tbody>
                <tr><td>dataset_path</td><td>str</td><td>Path to dataset directory</td></tr>
                <tr><td>solver_type</td><td>str</td><td><code>"anacp"</code>, <code>"fly"</code>, or <code>"wisard"</code></td></tr>
                <tr><td>**solver_kwargs</td><td>dict</td><td><code>expansion_ratio</code> (FlyCL), <code>num_bits</code> (WiSARD)</td></tr>
              </tbody>
            </table>
            <p><strong>Returns:</strong> dict with keys: <code>fingerprint</code>, <code>selected_backbone</code>, <code>solver</code>, <code>num_samples</code>, <code>class_names</code></p>
          </div>

          <div className="api-method">
            <h4><code>predict_image(image_path, model_context)</code></h4>
            <p>Predict the class of a single image.</p>
            <table className="param-table">
              <thead><tr><th>Parameter</th><th>Type</th><th>Description</th></tr></thead>
              <tbody>
                <tr><td>image_path</td><td>str</td><td>Path to the image file</td></tr>
                <tr><td>model_context</td><td>dict</td><td>The loaded model dictionary</td></tr>
              </tbody>
            </table>
            <p><strong>Returns:</strong> str — predicted class name</p>
          </div>
        </div>
      </section>

      {/* Config */}
      <section id="config">
        <h2>vyntri.core.config</h2>
        <div className="api-class">
          <h3><code>Config</code></h3>
          <p>Global configuration dataclass. All fields have sensible defaults.</p>
          <table className="param-table">
            <thead><tr><th>Field</th><th>Default</th><th>Description</th></tr></thead>
            <tbody>
              <tr><td>use_gpu_if_available</td><td>False</td><td>CPU-first philosophy. Set True for GPU.</td></tr>
              <tr><td>fingerprint_sample_size</td><td>1000</td><td>Max images sampled for fingerprinting</td></tr>
              <tr><td>pvi_backbone</td><td>"mobilenet_v3_small"</td><td>Backbone used for PVI estimation</td></tr>
              <tr><td>pvi_batch_size</td><td>32</td><td>Batch size for PVI feature extraction</td></tr>
              <tr><td>num_workers</td><td>4</td><td>Number of data loading workers</td></tr>
              <tr><td>cache_dir</td><td>".vyntri_cache"</td><td>Directory for cached data</td></tr>
            </tbody>
          </table>
          <CodeBlock title="Example">{`from vyntri.core.config import Config

config = Config(
    use_gpu_if_available=True,
    fingerprint_sample_size=500
)
`}</CodeBlock>
        </div>
      </section>

      {/* IO */}
      <section id="io">
        <h2>vyntri.core.io</h2>

        <div className="api-method">
          <h4><code>save_model(pipeline_result, path)</code></h4>
          <p>Save a trained model to disk using <code>torch.save</code>.</p>
          <table className="param-table">
            <thead><tr><th>Parameter</th><th>Type</th><th>Description</th></tr></thead>
            <tbody>
              <tr><td>pipeline_result</td><td>dict</td><td>Return value from <code>pipeline.run()</code></td></tr>
              <tr><td>path</td><td>str</td><td>File path to save (e.g. <code>"model.pkl"</code>)</td></tr>
            </tbody>
          </table>
        </div>

        <div className="api-method">
          <h4><code>load_model(path)</code></h4>
          <p>Load a saved model. Reconstructs the backbone and solver for inference.</p>
          <p><strong>Returns:</strong> dict with keys: <code>backbone</code>, <code>backbone_name</code>, <code>solver</code>, <code>classes</code>, <code>fingerprint</code>, <code>class_names</code></p>
          <div className="warning-box">
            <strong>Security:</strong> Only load models from trusted sources. Uses pickle internally.
          </div>
        </div>
      </section>

      {/* Dataset Engine */}
      <section id="engine">
        <h2>vyntri.dataset.engine</h2>
        <div className="api-class">
          <h3><code>DatasetIntelligenceEngine(config=None)</code></h3>
          <div className="api-method">
            <h4><code>analyze(dataset_path)</code></h4>
            <p>Performs comprehensive dataset analysis including statistics, visual fingerprint, PVI complexity, and quality checks.</p>
            <p><strong>Returns:</strong> dict with keys: <code>stats</code>, <code>visual</code>, <code>complexity</code>, <code>quality</code></p>
          </div>
        </div>
      </section>

      {/* PVI */}
      <section id="pvi">
        <h2>vyntri.dataset.pvi</h2>
        <div className="api-class">
          <h3><code>PVIEstimator(config)</code></h3>
          <p>Estimates Projection Variability Index (PVI) — measures how much a pre-trained backbone already "knows" about the data.</p>
          <div className="api-method">
            <h4><code>compute_pvi(dataset_path)</code></h4>
            <p><strong>Returns:</strong> dict with <code>mean_pvi</code>, <code>std_pvi</code>, <code>min_pvi</code>, <code>max_pvi</code>, <code>p5</code>, <code>p95</code></p>
          </div>
        </div>
      </section>

      {/* Backbones */}
      <section id="backbones">
        <h2>vyntri.backbones.loader</h2>
        <div className="api-method">
          <h4><code>load_backbone(backbone_name, pretrained=True)</code></h4>
          <p>Load a pre-trained backbone with the classification head removed.</p>
          <table className="param-table">
            <thead><tr><th>Backbone</th><th>Output Dim</th><th>Speed</th><th>Best For</th></tr></thead>
            <tbody>
              <tr><td>mobilenet_v3_small</td><td>576</td><td>Fast</td><td>Edge devices, simple tasks</td></tr>
              <tr><td>resnet18</td><td>512</td><td>Medium</td><td>Hard tasks, many classes</td></tr>
            </tbody>
          </table>
        </div>
        <div className="api-method">
          <h4><code>get_transform(input_size=224)</code></h4>
          <p>Returns standard ImageNet preprocessing transforms (Resize → CenterCrop → ToTensor → Normalize).</p>
        </div>
      </section>

      {/* Solvers */}
      <section id="solvers">
        <h2>vyntri.solvers</h2>
        <p>See the full <a href="#/solvers">Solvers Guide</a> for detailed comparisons and usage.</p>
        <table className="param-table">
          <thead><tr><th>Solver</th><th>Import</th></tr></thead>
          <tbody>
            <tr><td>AnaCP</td><td><code>from vyntri.solvers import AnalyticContrastiveProjection</code></td></tr>
            <tr><td>Continual AnaCP</td><td><code>from vyntri.solvers import ContinualAnaCP</code></td></tr>
            <tr><td>FlyCL</td><td><code>from vyntri.solvers import FlyCL</code></td></tr>
            <tr><td>WiSARD</td><td><code>from vyntri.solvers import WiSARD</code></td></tr>
          </tbody>
        </table>
      </section>
    </div>
  )
}

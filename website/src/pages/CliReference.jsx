import CodeBlock from '../components/CodeBlock'

export default function CliReference() {
  return (
    <div className="doc-page">
      <h1>CLI Reference</h1>
      <p className="lead">
        Vyntri installs a <code>vyntri</code> command with four sub-commands:
        <strong> analyze</strong>, <strong>train</strong>, <strong>predict</strong>, and <strong>update</strong>.
      </p>

      {/* Analyze */}
      <section>
        <h2 id="analyze">vyntri analyze</h2>
        <p>Inspect a dataset without training: class distribution, image statistics, and PVI complexity score.</p>
        <CodeBlock title="Usage">{`vyntri analyze [DATASET_PATH]`}</CodeBlock>
        <table className="param-table">
          <thead><tr><th>Argument</th><th>Default</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td>DATASET_PATH</td><td>./dataset</td><td>Path to folder-per-class dataset</td></tr>
          </tbody>
        </table>
        <CodeBlock title="Example">{`$ vyntri analyze ./cifar10

──────────────────────────────────
  Dataset Analysis
──────────────────────────────────
  Classes:           10
  Total images:      500
  Avg images/class:  50
  PVI complexity:    0.42
  Recommended:       mobilenet_v3_small
──────────────────────────────────`}</CodeBlock>
      </section>

      {/* Train */}
      <section>
        <h2 id="train">vyntri train</h2>
        <p>Run the full pipeline — analyze, select backbone, extract features, fit solver, save model.</p>
        <CodeBlock title="Usage">{`vyntri train [DATASET_PATH] [OPTIONS]`}</CodeBlock>
        <table className="param-table">
          <thead><tr><th>Option</th><th>Default</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td>DATASET_PATH</td><td>./dataset</td><td>Path to folder-per-class dataset</td></tr>
            <tr><td>--solver</td><td>anacp</td><td>Solver type: <code>anacp</code>, <code>fly</code>, or <code>wisard</code></td></tr>
            <tr><td>--save_path</td><td>model.pkl</td><td>Output model file path</td></tr>
          </tbody>
        </table>
        <CodeBlock title="Example">{`$ vyntri train ./dataset --solver fly --save_path my_model.pkl

[1/4] Analyzing dataset...
[2/4] Selecting backbone: mobilenet_v3_small
[3/4] Extracting features...
[4/4] Fitting FlyCL solver...

Model saved to my_model.pkl
  Solver:   fly
  Classes:  10
  Samples:  500`}</CodeBlock>
      </section>

      {/* Predict */}
      <section>
        <h2 id="predict">vyntri predict</h2>
        <p>Load a saved model and classify an image.</p>
        <CodeBlock title="Usage">{`vyntri predict [IMAGE_PATH] [MODEL_PATH]`}</CodeBlock>
        <table className="param-table">
          <thead><tr><th>Argument</th><th>Default</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td>IMAGE_PATH</td><td><em>required</em></td><td>Path to the image to classify</td></tr>
            <tr><td>MODEL_PATH</td><td>model.pkl</td><td>Path to saved model file</td></tr>
          </tbody>
        </table>
        <CodeBlock title="Example">{`$ vyntri predict photo.jpg my_model.pkl

Prediction: cat`}</CodeBlock>
      </section>

      {/* Update */}
      <section>
        <h2 id="update">vyntri update</h2>
        <p>
          Incrementally update an existing model with new data using <strong>ContinualAnaCP</strong>.
          The original training data is not needed — only the saved model and the new data.
        </p>
        <CodeBlock title="Usage">{`vyntri update [NEW_DATA_PATH] [MODEL_PATH]`}</CodeBlock>
        <table className="param-table">
          <thead><tr><th>Argument</th><th>Default</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td>NEW_DATA_PATH</td><td><em>required</em></td><td>Path to new folder-per-class data</td></tr>
            <tr><td>MODEL_PATH</td><td>model.pkl</td><td>Path to existing model (updated in-place)</td></tr>
          </tbody>
        </table>
        <CodeBlock title="Example">{`$ vyntri update ./new_images model.pkl

Loading model from model.pkl...
Extracting features from new data...
Updating solver...

Updated model saved to model.pkl
  New samples added:  120
  Total classes now:   12`}</CodeBlock>
      </section>

      {/* Tip */}
      <section>
        <div className="tip-box">
          <strong>Tip:</strong> All CLI commands validate paths before running. If a dataset directory doesn't exist or contains no class folders, you'll get a clear error message immediately.
        </div>
      </section>
    </div>
  )
}

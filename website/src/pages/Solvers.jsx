import CodeBlock from '../components/CodeBlock'

export default function Solvers() {
  return (
    <div className="doc-page">
      <h1>Solvers Guide</h1>
      <p className="lead">
        Vyntri ships three analytic solvers — zero gradient descent, zero epochs, zero hyperparameter sweeps.
        Each one solves a different sweet-spot of dataset size, speed, and memory.
      </p>

      {/* Comparison Table */}
      <section>
        <h2>Quick Comparison</h2>
        <div className="table-wrap">
          <table className="param-table">
            <thead>
              <tr><th></th><th>AnaCP</th><th>FlyCL</th><th>WiSARD</th></tr>
            </thead>
            <tbody>
              <tr><td><strong>Full Name</strong></td><td>Analytic Contrastive Projection</td><td>Fly-Inspired Continual Learner</td><td>Weightless Neural Network</td></tr>
              <tr><td><strong>How it works</strong></td><td>LDA projection + Ridge regression</td><td>Bio-inspired sparse coding + hash tables</td><td>Thermometer encoding + RAM lookups</td></tr>
              <tr><td><strong>Speed</strong></td><td>Fast</td><td>Fastest</td><td>Fast</td></tr>
              <tr><td><strong>Memory</strong></td><td>Low</td><td>Very low</td><td>Low</td></tr>
              <tr><td><strong>Continual learning?</strong></td><td>Yes (ContinualAnaCP)</td><td>Built-in</td><td>Built-in</td></tr>
              <tr><td><strong>Best for</strong></td><td>General classification</td><td>Tiny datasets, streaming data</td><td>Binary-friendly features, edge hardware</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* AnaCP */}
      <section>
        <h2 id="anacp">AnaCP — Analytic Contrastive Projection</h2>
        <div className="solver-detail">
          <h3>How It Works</h3>
          <ol>
            <li><strong>LDA Projection</strong> — computes between-class and within-class scatter matrices, then projects features into a space that maximizes class separation.</li>
            <li><strong>Ridge Regression</strong> — fits a linear classifier in the projected space using a closed-form solve: <code>(X<sup>T</sup>X + αI)<sup>-1</sup>X<sup>T</sup>y</code>.</li>
          </ol>
          <h3>Parameters</h3>
          <table className="param-table">
            <thead><tr><th>Parameter</th><th>Default</th><th>Description</th></tr></thead>
            <tbody>
              <tr><td>alpha</td><td>1.0</td><td>Ridge regularization strength</td></tr>
            </tbody>
          </table>
          <CodeBlock title="Python">{`from vyntri.core.pipeline import VyntriPipeline

pipeline = VyntriPipeline()
result = pipeline.run("./dataset", solver_type="anacp")
# AnaCP is the default solver`}</CodeBlock>
        </div>
      </section>

      {/* ContinualAnaCP */}
      <section>
        <h2 id="continual">ContinualAnaCP — Incremental Updates</h2>
        <div className="solver-detail">
          <h3>How It Works</h3>
          <p>
            Same math as AnaCP, but maintains running sufficient statistics (<code>X<sup>T</sup>X</code>, <code>X<sup>T</sup>y</code>, per-class means & counts).
            When new data arrives, statistics are updated and the solver re-computes the closed-form solution without needing the original data.
          </p>
          <h3>Parameters</h3>
          <table className="param-table">
            <thead><tr><th>Parameter</th><th>Default</th><th>Description</th></tr></thead>
            <tbody>
              <tr><td>alpha</td><td>1.0</td><td>Ridge regularization strength</td></tr>
            </tbody>
          </table>
          <CodeBlock title="Continual Learning">{`from vyntri.solvers import ContinualAnaCP

# Train on initial features
solver = ContinualAnaCP(alpha=1.0)
solver.fit(features_batch1, labels_batch1)

# Later, update with new data (Recursive Least Squares)
solver.update(features_batch2, labels_batch2)`}</CodeBlock>
        </div>
      </section>

      {/* FlyCL */}
      <section>
        <h2 id="fly">FlyCL — Fly-Inspired Continual Learner</h2>
        <div className="solver-detail">
          <h3>How It Works</h3>
          <p>Inspired by the fruit fly olfactory circuit:</p>
          <ol>
            <li><strong>Random Expansion</strong> — feature vectors are projected into a high-dimensional sparse space via a random matrix (analogous to Kenyon cells).</li>
            <li><strong>Winner-Take-All</strong> — only the top-k activations survive (sparse hash).</li>
            <li><strong>Hash-Table Lookup</strong> — training stores (hash → label) pairs; prediction uses majority vote.</li>
          </ol>
          <h3>Parameters</h3>
          <table className="param-table">
            <thead><tr><th>Parameter</th><th>Default</th><th>Description</th></tr></thead>
            <tbody>
              <tr><td>expansion_ratio</td><td>10</td><td>Expands features into <code>dim × ratio</code> hash space</td></tr>
              <tr><td>sparsity</td><td>0.05</td><td>Fraction of activations kept (winner-take-all)</td></tr>
              <tr><td>seed</td><td>42</td><td>Random seed for reproducibility</td></tr>
            </tbody>
          </table>
          <CodeBlock title="Python">{`result = pipeline.run("./dataset", solver_type="fly",
                      expansion_ratio=20)`}</CodeBlock>
        </div>
      </section>

      {/* WiSARD */}
      <section>
        <h2 id="wisard">WiSARD — Weightless Neural Network</h2>
        <div className="solver-detail">
          <h3>How It Works</h3>
          <ol>
            <li><strong>Thermometer Encoding</strong> — continuous features are binarized into thermometer codes (e.g., 0.7 with 8 bits → <code>11111100</code>).</li>
            <li><strong>RAM Discriminators</strong> — groups of bits address lookup tables (RAMs). Each class has its own set of RAMs.</li>
            <li><strong>Response Counting</strong> — prediction sums the RAM responses for each class. Highest sum wins.</li>
          </ol>
          <h3>Parameters</h3>
          <table className="param-table">
            <thead><tr><th>Parameter</th><th>Default</th><th>Description</th></tr></thead>
            <tbody>
              <tr><td>num_bits</td><td>8</td><td>Bits per thermometer code (quantization resolution)</td></tr>
              <tr><td>address_size</td><td>4</td><td>Number of bits grouped per RAM address</td></tr>
            </tbody>
          </table>
          <CodeBlock title="Python">{`result = pipeline.run("./dataset", solver_type="wisard",
                      num_bits=16)`}</CodeBlock>
        </div>
      </section>

      {/* When to Use What */}
      <section>
        <h2>When To Use Which Solver</h2>
        <div className="solver-scenarios">
          <div className="scenario-card">
            <h3>General Classification</h3>
            <p>Use <strong>AnaCP</strong> — best overall accuracy via LDA projection.</p>
          </div>
          <div className="scenario-card">
            <h3>Data Arriving Over Time</h3>
            <p>Use <strong>ContinualAnaCP</strong> — updates without retraining from scratch.</p>
          </div>
          <div className="scenario-card">
            <h3>Very Few Samples</h3>
            <p>Use <strong>FlyCL</strong> — sparse hashing works well with minimal data.</p>
          </div>
          <div className="scenario-card">
            <h3>Ultra-Low Compute</h3>
            <p>Use <strong>WiSARD</strong> — no floating-point math at inference, only RAM lookups.</p>
          </div>
        </div>
      </section>
    </div>
  )
}

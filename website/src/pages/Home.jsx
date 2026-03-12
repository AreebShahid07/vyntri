import { Link } from 'react-router-dom'
import CodeBlock from '../components/CodeBlock'

export default function Home() {
  return (
    <div className="page">
      {/* Hero */}
      <section className="hero">
        <div className="hero-badge">v0.1.0 — Now on PyPI</div>
        <h1 className="hero-title">
          Training-Less Vision Intelligence<br />
          <span className="hero-accent">for Edge Devices</span>
        </h1>
        <p className="hero-subtitle">
          Vyntri replaces gradient descent with closed-form linear algebra.
          Train image classifiers in milliseconds — no GPUs, no epochs, no backpropagation.
          Built for Raspberry Pi, CPU-only servers, and real-time embedded systems.
        </p>
        <div className="hero-actions">
          <Link to="/getting-started" className="btn btn-primary">Get Started</Link>
          <a
            href="https://github.com/AreebShahid07/vyntri"
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-secondary"
          >
            View on GitHub
          </a>
        </div>
        <CodeBlock title="Install">pip install vyntri</CodeBlock>
      </section>

      {/* Features */}
      <section className="features">
        <h2 className="section-title">Why Vyntri?</h2>
        <div className="feature-grid">
          <div className="feature-card">
            <div className="feature-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
            </div>
            <h3>Instant Training</h3>
            <p>No epochs, no loss curves. Analytic solvers compute optimal weights in a single pass using closed-form solutions.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M15 2v2M15 20v2M2 15h2M20 15h2M2 9h2M20 9h2M9 2v2M9 20v2"/></svg>
            </div>
            <h3>Edge-First</h3>
            <p>CPU by default. MobileNetV3 backbone. Designed for Raspberry Pi, Jetson Nano, and low-power hardware.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
            </div>
            <h3>Continual Learning</h3>
            <p>Add new classes without retraining. Recursive Least Squares updates the model incrementally — no catastrophic forgetting.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>
            </div>
            <h3>Folder-per-Class</h3>
            <p>No config files, no annotations. Just organize images into folders named by class and point Vyntri at them.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93A10 10 0 0 0 4.93 19.07M12 2v2M12 20v2M2 12h2M20 12h2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>
            </div>
            <h3>3 Solver Algorithms</h3>
            <p>AnaCP (LDA + Ridge), FlyCL (bio-inspired sparse coding), and WiSARD (weightless neural network). Pick the best fit.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>
            </div>
            <h3>CLI + Python API</h3>
            <p>Full command-line interface for analyze, train, predict, and update. Plus a clean Python API for integration.</p>
          </div>
        </div>
      </section>

      {/* Quick Example */}
      <section className="quick-example">
        <h2 className="section-title">Quick Example</h2>
        <div className="example-grid">
          <div className="example-col">
            <h3>Python API</h3>
            <CodeBlock title="train_and_predict.py">{`from vyntri.core.pipeline import VyntriPipeline
from vyntri.core.config import Config
from vyntri.core.io import save_model, load_model

# Train (analyzes dataset, selects backbone, fits solver)
pipeline = VyntriPipeline(Config())
result = pipeline.run("./my_dataset")

# Save
save_model(result, "model.pkl")

# Predict
model = load_model("model.pkl")
prediction = pipeline.predict_image("test.jpg", model)
print(prediction)  # "cat"`}</CodeBlock>
          </div>
          <div className="example-col">
            <h3>CLI</h3>
            <CodeBlock title="Terminal">{`# Analyze your dataset
vyntri analyze ./my_dataset

# Train a model
vyntri train ./my_dataset --save_path model.pkl --solver anacp

# Predict an image
vyntri predict photo.jpg model.pkl

# Add new data (continual learning)
vyntri update ./new_data model.pkl`}</CodeBlock>
          </div>
        </div>
      </section>

      {/* Architecture */}
      <section className="architecture">
        <h2 className="section-title">How It Works</h2>
        <div className="pipeline-steps">
          <div className="pipeline-step">
            <div className="step-number">1</div>
            <h3>Analyze</h3>
            <p>Dataset intelligence engine fingerprints your data — class balance, resolution, PVI complexity scoring.</p>
          </div>
          <div className="pipeline-arrow">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
          </div>
          <div className="pipeline-step">
            <div className="step-number">2</div>
            <h3>Select</h3>
            <p>Zero-cost model selection picks the optimal backbone (MobileNetV3 or ResNet18) based on dataset difficulty.</p>
          </div>
          <div className="pipeline-arrow">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
          </div>
          <div className="pipeline-step">
            <div className="step-number">3</div>
            <h3>Extract</h3>
            <p>Frozen pre-trained backbone extracts feature vectors from your images. No fine-tuning needed.</p>
          </div>
          <div className="pipeline-arrow">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
          </div>
          <div className="pipeline-step">
            <div className="step-number">4</div>
            <h3>Adapt</h3>
            <p>Analytic solver computes optimal classifier weights in one shot via closed-form linear algebra.</p>
          </div>
        </div>
      </section>

      {/* Dataset Format */}
      <section className="dataset-format">
        <h2 className="section-title">Dataset Format</h2>
        <p className="section-desc">Vyntri expects a simple folder-per-class structure. No YAML configs, no annotation files.</p>
        <CodeBlock title="Expected structure">{`my_dataset/
├── cat/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── dog/
│   ├── img001.jpg
│   └── ...
└── bird/
    ├── img001.jpg
    └── ...`}</CodeBlock>
        <p className="section-note">Supports JPG, JPEG, PNG, BMP, TIFF, and WebP formats.</p>
      </section>
    </div>
  )
}

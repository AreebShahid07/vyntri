export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-inner">
        <div className="footer-col">
          <h4>Vyntri</h4>
          <p>Training-less vision intelligence for edge devices.</p>
        </div>
        <div className="footer-col">
          <h4>Links</h4>
          <a href="https://pypi.org/project/vyntri/" target="_blank" rel="noopener noreferrer">PyPI Package</a>
          <a href="https://github.com/AreebShahid07/vyntri" target="_blank" rel="noopener noreferrer">GitHub Repository</a>
          <a href="https://github.com/AreebShahid07/vyntri/issues" target="_blank" rel="noopener noreferrer">Report an Issue</a>
        </div>
        <div className="footer-col">
          <h4>Install</h4>
          <code className="footer-code">pip install vyntri</code>
        </div>
      </div>
      <div className="footer-bottom">
        <p>&copy; {new Date().getFullYear()} Vyntri Contributors. MIT License.</p>
      </div>
    </footer>
  )
}

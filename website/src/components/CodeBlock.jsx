import { useState } from 'react'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'

/* ── Icons ──────────────────────────────────────────── */
const CheckIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12" />
  </svg>
)
const CopyIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
  </svg>
)
const TerminalIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="4 17 10 11 4 5" /><line x1="12" y1="19" x2="20" y2="19" />
  </svg>
)
const FileIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
    <polyline points="14 2 14 8 20 8" />
  </svg>
)
const CodeIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="16 18 22 12 16 6" /><polyline points="8 6 2 12 8 18" />
  </svg>
)

/* ── Language detection ─────────────────────────────── */
function detectLang(title) {
  if (!title) return 'text'
  const t = title.toLowerCase()
  if (t.includes('terminal') || t.includes('install') || t.includes('usage') || t.includes('example')) return 'bash'
  if (t.endsWith('.py') || t.includes('python') || t.includes('continual')) return 'python'
  if (t.includes('folder') || t.includes('structure')) return 'text'
  return 'python'
}

function detectIcon(title) {
  if (!title) return <CodeIcon />
  const t = title.toLowerCase()
  if (t.includes('terminal') || t.includes('usage') || t.includes('example') || t.includes('install')) return <TerminalIcon />
  if (t.endsWith('.py') || t.includes('python') || t.includes('continual') || t.includes('.py')) return <FileIcon />
  return <CodeIcon />
}

/* ── Custom Prism theme ─────────────────────────────── */
const vyntriTheme = {
  'code[class*="language-"]': {
    color: '#CBD5E1',
    background: 'none',
    fontFamily: "'JetBrains Mono', ui-monospace, 'Cascadia Code', monospace",
    fontSize: '0.85rem',
    lineHeight: '1.8',
  },
  'pre[class*="language-"]': {
    color: '#CBD5E1',
    background: 'none',
    margin: 0,
    padding: '1.25rem',
    overflow: 'auto',
  },
  // comments
  comment:    { color: '#4a5568', fontStyle: 'italic' },
  prolog:     { color: '#4a5568' },
  doctype:    { color: '#4a5568' },
  cdata:      { color: '#4a5568' },
  // punctuation
  punctuation: { color: '#8892a8' },
  // Python keywords → blue
  keyword:    { color: '#60A5FA', fontWeight: '600' },
  // strings → amber/yellow
  string:     { color: '#FCD34D' },
  // numbers → cyan
  number:     { color: '#67E8F9' },
  // built-ins / class names → purple
  'class-name': { color: '#C084FC' },
  builtin:    { color: '#C084FC' },
  // functions → green
  function:   { color: '#34D399' },
  // operators → orange
  operator:   { color: '#FB923C' },
  // booleans / constants → pink
  boolean:    { color: '#F472B6' },
  constant:   { color: '#F472B6' },
  // imports
  'maybe-class-name': { color: '#C084FC' },
  // namespace
  namespace:  { color: '#94A3B8' },
  // bash specifics
  'function-name': { color: '#34D399' },
  parameter:  { color: '#CBD5E1' },
  variable:   { color: '#FB923C' },
  property:   { color: '#60A5FA' },
}

/* ── Bash: make $ prompt dim, flags highlighted ─────── */
function formatBashCode(code) {
  return code // Prism handles bash natively
}

/* ── Component ──────────────────────────────────────── */
export default function CodeBlock({ children, title }) {
  const [copied, setCopied] = useState(false)
  const lang = detectLang(title)
  const isTerminal = lang === 'bash'
  const isPlainText = lang === 'text'
  const code = typeof children === 'string' ? children : String(children ?? '')

  const handleCopy = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  return (
    <div className={`code-block${isTerminal ? ' code-block--terminal' : ''}`}>
      {/* Header */}
      <div className="code-block-header">
        <div className="code-block-left">
          {isTerminal && (
            <div className="terminal-dots">
              <span className="dot dot-red" />
              <span className="dot dot-yellow" />
              <span className="dot dot-green" />
            </div>
          )}
          {title && (
            <div className="code-block-title">
              <span className="code-block-title-icon">{detectIcon(title)}</span>
              {title}
            </div>
          )}
        </div>
        <button
          className={`copy-btn${copied ? ' copy-btn--copied' : ''}`}
          onClick={handleCopy}
          aria-label="Copy code"
          title="Copy to clipboard"
        >
          {copied ? <CheckIcon /> : <CopyIcon />}
          <span>{copied ? 'Copied!' : 'Copy'}</span>
        </button>
      </div>

      {/* Code body */}
      {isPlainText ? (
        <pre className="code-block-plain"><code>{code}</code></pre>
      ) : (
        <SyntaxHighlighter
          language={lang}
          style={vyntriTheme}
          PreTag="div"
          customStyle={{
            margin: 0,
            padding: '1.25rem',
            background: 'transparent',
            fontSize: '0.85rem',
            lineHeight: '1.8',
            overflowX: 'auto',
          }}
          codeTagProps={{
            style: {
              fontFamily: "'JetBrains Mono', ui-monospace, 'Cascadia Code', monospace",
            }
          }}
        >
          {code}
        </SyntaxHighlighter>
      )}
    </div>
  )
}

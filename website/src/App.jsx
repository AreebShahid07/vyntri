import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom'
import { useEffect } from 'react'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Home from './pages/Home'
import GettingStarted from './pages/GettingStarted'
import ApiReference from './pages/ApiReference'
import Solvers from './pages/Solvers'
import CliReference from './pages/CliReference'
import DocLayout from './components/DocLayout'
import './App.css'
import { Analytics } from '@vercel/analytics/react';
import { SpeedInsights } from '@vercel/speed-insights/react';

const SITE_URL = 'https://www.vyntri.dev'

const SEO_BY_PATH = {
  '/': {
    title: 'Vyntri | Training-Less Vision Intelligence for Edge AI',
    description: 'Vyntri is a fast, training-less vision intelligence library for edge AI. Build image classification pipelines in milliseconds on CPU-only devices.'
  },
  '/getting-started': {
    title: 'Getting Started | Vyntri Docs',
    description: 'Get started with Vyntri in minutes. Install the package, load your dataset, and run CPU-first training-less vision workflows.'
  },
  '/api': {
    title: 'API Reference | Vyntri Docs',
    description: 'Explore the Vyntri API reference for pipeline components, dataset utilities, and edge-ready vision building blocks.'
  },
  '/solvers': {
    title: 'Solvers | Vyntri Docs',
    description: 'Compare Vyntri solvers and choose the right option for fast, training-less image intelligence on constrained hardware.'
  },
  '/cli': {
    title: 'CLI Reference | Vyntri Docs',
    description: 'Use the Vyntri CLI to run reproducible edge AI workflows from the command line with minimal setup.'
  }
}

function updateMetaTag(selector, value, attributeName = 'content') {
  const element = document.querySelector(selector)
  if (element) {
    element.setAttribute(attributeName, value)
  }
}

function RouteSeo() {
  const { pathname } = useLocation()

  useEffect(() => {
    const seo = SEO_BY_PATH[pathname] || SEO_BY_PATH['/']
    const canonicalUrl = pathname === '/' ? `${SITE_URL}/` : `${SITE_URL}${pathname}`

    document.title = seo.title
    updateMetaTag('meta[name="description"]', seo.description)
    updateMetaTag('meta[property="og:title"]', seo.title)
    updateMetaTag('meta[property="og:description"]', seo.description)
    updateMetaTag('meta[property="og:url"]', canonicalUrl)
    updateMetaTag('meta[name="twitter:title"]', seo.title)
    updateMetaTag('meta[name="twitter:description"]', seo.description)
    updateMetaTag('link[rel="canonical"]', canonicalUrl, 'href')
  }, [pathname])

  return null
}

function ScrollToTop() {
  const { pathname } = useLocation()
  useEffect(() => { window.scrollTo(0, 0) }, [pathname])
  return null
}

function App() {
  return (
    <Router>
      <ScrollToTop />
      <RouteSeo />
      <div className="app">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route element={<DocLayout />}>
              <Route path="/getting-started" element={<GettingStarted />} />
              <Route path="/api" element={<ApiReference />} />
              <Route path="/solvers" element={<Solvers />} />
              <Route path="/cli" element={<CliReference />} />
            </Route>
          </Routes>
        </main>
        <Footer />
      </div>
      <Analytics />
      <SpeedInsights />
    </Router>
  )
}

export default App

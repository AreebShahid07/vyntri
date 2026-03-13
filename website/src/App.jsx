import { HashRouter as Router, Routes, Route, useLocation } from 'react-router-dom'
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

function ScrollToTop() {
  const { pathname } = useLocation()
  useEffect(() => { window.scrollTo(0, 0) }, [pathname])
  return null
}

function App() {
  return (
    <Router>
      <ScrollToTop />
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

import { HashRouter as Router, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Home from './pages/Home'
import GettingStarted from './pages/GettingStarted'
import ApiReference from './pages/ApiReference'
import Solvers from './pages/Solvers'
import CliReference from './pages/CliReference'
import DocLayout from './components/DocLayout'
import './App.css'

function App() {
  return (
    <Router>
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
    </Router>
  )
}

export default App

import "./App.css";
import About from "./pages/About";
import Dataset from "./pages/Dataset";
import Analyze from "./pages/Analyze";
import Features from "./components/Features";
import Navbar from "./components/Navbar";
import Hero from "./components/Hero";

function App() {
  return (
    <div className="app">

      <Navbar />

      <Hero />

<div className="hero-button-container">
 <a href="#analyze" className="primary-button">
  ANALYZE MEDICAL IMAGE →
</a>
</div>
<Features />
<Analyze />
<Dataset />
<About />
    </div>
  );
}

export default App;
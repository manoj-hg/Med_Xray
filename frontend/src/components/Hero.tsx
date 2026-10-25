function Hero() {
  return (
    <section className="hero" id="home">

      {/* Left side */}
      <div className="hero-text hero-left">
        <h1>medical</h1>

        <p>
          Intelligent medical image analysis
          powered by artificial intelligence.
        </p>
      </div>


      {/* Center */}
      <div className="hero-image">

        <div className="medical-image">
          <div className="scan-circle"></div>

          <div className="scan-text">
            AI
          </div>

          <span>MEDICAL</span>
          <span>VISION</span>
        </div>

      </div>


      {/* Right side */}
      <div className="hero-text hero-right">
        <h1>&lt;AI&gt;</h1>

        <p>
          Analyze medical images, generate
          descriptions and identify concepts.
        </p>
      </div>

    </section>
  );
}

export default Hero;
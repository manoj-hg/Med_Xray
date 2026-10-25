function Features() {
  return (
    <section className="features" id="features">

      <div className="features-heading">
        <p>EXPLORE OUR CAPABILITIES</p>
        <h2>Medical AI Intelligence</h2>
      </div>


      <div className="feature-grid">

        <div className="feature-card">
          <span className="feature-number">01</span>

          <h3>IMAGE ANALYSIS</h3>

          <p>
            Upload medical images and analyze
            them using artificial intelligence
            and computer vision.
          </p>

          <span className="feature-arrow">↗</span>
        </div>


        <div className="feature-card">
          <span className="feature-number">02</span>

          <h3>AI CAPTIONING</h3>

          <p>
            Generate meaningful medical
            descriptions from uploaded
            images using AI models.
          </p>

          <span className="feature-arrow">↗</span>
        </div>


        <div className="feature-card">
          <span className="feature-number">03</span>

          <h3>MEDICAL CONCEPTS</h3>

          <p>
            Identify medical concepts and
            connect concept identifiers with
            readable medical terminology.
          </p>

          <span className="feature-arrow">↗</span>
        </div>

      </div>

    </section>
  );
}

export default Features;
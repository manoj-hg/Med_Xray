function About() {
  return (
    <section className="about-page" id="about">

      <div className="about-heading">
        <p>ABOUT THE PROJECT</p>

        <h1>
          Intelligence for
          <br />
          Medical Imaging
        </h1>

        <span>
          A medical AI platform designed to analyze images,
          generate meaningful descriptions and identify
          important medical concepts.
        </span>
      </div>


      <div className="about-content">

        <div className="about-number">
          01
        </div>

        <div className="about-text">

          <h2>
            Making Medical Images
            <br />
            Easier to Understand
          </h2>

          <p>
            Medical image analysis can help healthcare professionals
            understand important visual information from medical
            images. This project uses artificial intelligence to
            analyze uploaded images and generate useful information
            from them.
          </p>

          <p>
            The system combines image analysis, AI-generated
            captions and medical concept identification to provide
            structured results. Medical concepts can also be mapped
            to standardized CUI identifiers for better organization
            of medical information.
          </p>

        </div>

      </div>


      <div className="about-features">

        <div>
          <span>01</span>
          <h3>IMAGE ANALYSIS</h3>
          <p>
            Analyze medical images using an AI-powered model.
          </p>
        </div>

        <div>
          <span>02</span>
          <h3>AI CAPTIONING</h3>
          <p>
            Generate meaningful descriptions from medical images.
          </p>
        </div>

        <div>
          <span>03</span>
          <h3>CONCEPT MAPPING</h3>
          <p>
            Identify medical concepts and map them to CUI terms.
          </p>
        </div>

      </div>

    </section>
  );
}

export default About;
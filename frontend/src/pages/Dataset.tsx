function Dataset() {
  return (
    <section className="dataset-page" id="dataset">

      <div className="dataset-heading">
        <p>MEDICAL AI DATASET</p>

        <h1>Dataset Overview</h1>

        <span>
          A structured collection of medical images,
          captions and medical concepts used for AI analysis.
        </span>
      </div>

      <div className="dataset-stats">

        <div className="dataset-stat">
          <strong>03</strong>
          <span>IMAGE SPLITS</span>
          <p>Training, validation and testing datasets.</p>
        </div>

        <div className="dataset-stat">
          <strong>03</strong>
          <span>DATA TYPES</span>
          <p>Images, captions and medical concepts.</p>
        </div>

        <div className="dataset-stat">
          <strong>CUI</strong>
          <span>MEDICAL MAPPING</span>
          <p>Medical identifiers mapped to readable terms.</p>
        </div>

      </div>


      <div className="dataset-files">

        <h2>Dataset Components</h2>

        <div className="dataset-file-grid">

          <div className="dataset-file">
            <span>01</span>
            <h3>TRAINING DATA</h3>
            <p>
              Medical images and associated captions and
              concepts used to train AI models.
            </p>
            <small>
              train_images.zip · train_captions.csv · train_concepts.csv
            </small>
          </div>


          <div className="dataset-file">
            <span>02</span>
            <h3>VALIDATION DATA</h3>
            <p>
              Images and annotations used to validate and
              tune model performance.
            </p>
            <small>
              valid_images.zip · valid_captions.csv · valid_concepts.csv
            </small>
          </div>


          <div className="dataset-file">
            <span>03</span>
            <h3>TEST DATA</h3>
            <p>
              Independent medical images used for final
              evaluation of the trained model.
            </p>
            <small>
              test_images.zip · test_captions.csv · test_concepts.csv
            </small>
          </div>


          <div className="dataset-file">
            <span>04</span>
            <h3>MANUAL CONCEPTS</h3>
            <p>
              Manually curated medical concepts associated
              with training, validation and test images.
            </p>
            <small>
              train · valid · test manual concepts
            </small>
          </div>


          <div className="dataset-file">
            <span>05</span>
            <h3>CUI MAPPING</h3>
            <p>
              Maps medical concept identifiers to readable
              medical terminology.
            </p>
            <small>
              cui_mapping.csv
            </small>
          </div>


          <div className="dataset-file">
            <span>06</span>
            <h3>LICENSE INFORMATION</h3>
            <p>
              Contains image-source, attribution and
              licensing information.
            </p>
            <small>
              license_information.csv
            </small>
          </div>

        </div>

      </div>

    </section>
  );
}

export default Dataset;
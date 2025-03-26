# Nationalist Sentiment Analysis: A Machine Learning Approach

This project addresses a recent controversy in sociology regarding nationalist sentiment in the United States. The following studies provide the context:

- [Bonikowski & DiMaggio (2016)](https://doi.org/10.1177/0003122416663683) used latent class analysis (LCA) on survey data to identify 4 varieties of American Nationalism.
- [Eger & Hjerm (2022)](https://doi.org/10.1111/nana.12722) reanalyzed the data and concluded that LCA was misinterpreted, and that the results do not allow distinguishing between different varieties of nationalist sentiment.
- [Bonikowski & DiMaggio (2022)](https://doi.org/10.1111/nana.12756) responded that their conclusion is supported by external criteria.

The aim of this project is to shed further light on this debate by using common clustering techniques from machine learning.

## Data

- The raw data must be downloaded from Bonikowski & DiMaggio's [replication package](https://journals.sagepub.com/doi/suppl/10.1177/0003122416663683/suppl_file/replication_package_online_supplements.zip).
- Transformed data is available in the repository.

## Code

- **1_prepare.Rmd**: Prepares the data (details of preprocessing steps).
- **2_desc_PCA.Rmd**: Performs descriptive analysis and principal component analysis (PCA).
- **3_clustering.ipynb**: Performs clustering using various functions stored in the `/src` directory.
- **model_eval**: Computes performance metrics for model evaluation.
- **model_fit**: Conducts the actual clustering process.
- **model_select**: Computes the Gap Statistic to select the optimal number of clusters.
- **hopkins**: Computes the Hopkins statistic to assess cluster tendency.
- **model_plot**: Generates plots for the clustering results.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


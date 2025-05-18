use ndarray::prelude::*;
use rustyml::prelude::*;
use polars::prelude::*;

fn csv_to_ndarray(path: &str) -> Array2<f64> {
    CsvReadOptions::default()
        .with_has_header(false)
        .try_into_reader_with_file_path(Some(path.into())).unwrap()
        .finish().unwrap()
        .to_ndarray::<Float64Type>(IndexOrder::C).unwrap()
}

// Add normalization function
fn normalize(arr: &Array2<f64>) -> Array2<f64> {
    let mean = arr.mean_axis(Axis(0)).unwrap();
    let std = arr.std_axis(Axis(0), 0.0);

    let mut normalized = arr.clone();
    for mut row in normalized.rows_mut() {
        for (i, val) in row.iter_mut().enumerate() {
            *val = (*val - mean[i]) / std[i];
        }
    }
    normalized
}

fn main() {
    let column_values: Vec<f64> = (0..755).map(|i| i as f64).collect();

    let x_original = Array::from_shape_vec((755, 1), column_values).unwrap();

    let data = csv_to_ndarray("Relative Sea Level Rise Kanmen, China.csv");

    let y = data.slice(s![.., 2]);

    println!("Original X data shape: {:?}", x_original.shape());
    println!("Original Y data shape: {:?}", y.shape());

    // Print some data statistics to help with diagnostics
    println!("X data range: [{}, {}]",
             x_original.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
             x_original.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );
    println!("Y data range: [{}, {}]",
             y.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
             y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    // Normalize X data
    let x = normalize(&x_original);

    // Use a smaller learning rate
    let mut model = LinearRegression::new(
        true,  // Use intercept
        0.001, // Lower learning rate
        10000, // Increase iterations to compensate for smaller learning rate
        1e-8,  // Stricter convergence condition
        None,
    );

    // Try to fit the model and handle possible errors
    match model.fit(x.view(), y) {
        Ok(_) => {
            println!("Model training completed successfully");
            // Use the trained model for prediction
            let y_pred = model.predict(x.view());
            println!("Prediction results: {:?}", y_pred);

            // Calculate and print R² value to evaluate model quality
            if let Ok(y_pred) = y_pred {
                let y_mean = y.mean().unwrap();
                let tss: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
                let rss: f64 = y.iter().zip(y_pred.iter())
                    .map(|(&yi, &y_hat_i)| (yi - y_hat_i).powi(2)).sum();
                let r_squared = 1.0 - (rss / tss);
                println!("R²: {}", r_squared);
            }

            let intercept = model.get_intercept().unwrap();
            let coefficients = model.get_coefficients().unwrap();

            // Calculate parameters in the original data scale
            let x_mean = x_original.mean_axis(Axis(0)).unwrap();
            let x_std = x_original.std_axis(Axis(0), 0.0);

            let original_coefficients: Vec<f64> = coefficients
                .iter()
                .enumerate()
                .map(|(i, &coef)| coef / x_std[i])
                .collect();

            let original_intercept = intercept - coefficients
                .iter()
                .enumerate()
                .map(|(i, &coef)| coef * x_mean[i] / x_std[i])
                .sum::<f64>();

            println!("Model intercept in original scale: {}", original_intercept);
            println!("Model coefficients in original scale: {:?}", original_coefficients);
            println!("\nSea level rise(mm) in Shanghai per year: {}", original_coefficients[0] * 1000.0 * 12.0);
        },
        Err(e) => {
            println!("Model training failed: {:?}", e);
        }
    }
}

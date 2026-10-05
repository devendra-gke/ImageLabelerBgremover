/*
===========================================================
 Gurukrupa Exports Pvt Ltd
 Background Removal Module (IMG.LY Official)
===========================================================
*/

let removeBackgroundFn = null;
let modelLoaded = false;

/**
 * Load AI model only once.
 */
export async function ensureModel() {

    if (modelLoaded && removeBackgroundFn) {
        return removeBackgroundFn;
    }

    const overlay = document.getElementById("modelOverlay");
    const title = document.getElementById("modelOverlayTitle");
    const text = document.getElementById("modelOverlayText");

    if (overlay) overlay.classList.add("is-visible");
    if (title) title.textContent = "Loading AI Background Removal Model...";
    if (text)
        text.textContent =
            "Please wait. The model is loading from your local Django server.";

    try {

        // Load official browser module
        const mod = await import("/static/bg-removal/browser.mjs");

        removeBackgroundFn = async (blob) => {

            return await mod.removeBackground(blob, {
                publicPath: "/static/bg-removal/models/",
                model: "isnet",

                output: {
                    format: "image/png",
                    quality: 1
                }
            });

        };

        modelLoaded = true;

        if (title) title.textContent = "AI Model Ready";
        if (text)
            text.textContent =
                "Background Removal model loaded successfully.";

        setTimeout(() => {
            if (overlay) overlay.classList.remove("is-visible");
        }, 500);

        console.log("✅ IMG.LY Background Removal Model Loaded");

        return removeBackgroundFn;

    } catch (err) {

        if (overlay) overlay.classList.remove("is-visible");

        console.error("Model Load Error:", err);

        throw new Error(
            "Unable to load Background Removal model.\n\n" + err.message
        );
    }
}

/**
 * Remove background from one image.
 * Returns PNG File.
 */
export async function removeImageBackground(file) {

    const removeBackground = await ensureModel();

    const pngBlob = await removeBackground(file);

    return new File(
        [pngBlob],
        file.name.replace(/\.[^.]+$/, ".png"),
        {
            type: "image/png"
        }
    );
}

/**
 * Upload processed PNG to Django.
 */
export async function uploadProcessedImage(file, meta) {

    const formData = new FormData();

    formData.append("image", file);
    formData.append("original_name", meta.originalName);
    formData.append("key", meta.key);
    formData.append("category", meta.category);
    formData.append("style", meta.style);

    formData.append(
        "itemwise_folder",
        document.getElementById("itemwiseFolder").value
    );

    formData.append(
        "stylebio_folder",
        document.getElementById("stylebioFolder").value
    );

    formData.append(
        "csrfmiddlewaretoken",
        document.querySelector("[name=csrfmiddlewaretoken]").value
    );

    const response = await fetch("/save-processed-image/", {
        method: "POST",
        body: formData
    });

    const result = await response.json();

    if (!response.ok) {
        throw new Error(result.error || "Upload failed.");
    }

    return result;
}

/**
 * Process multiple images sequentially.
 */
export async function processImages(imageFiles, excelMap, callbacks = {}) {

    const total = imageFiles.length;

    let completed = 0;

    for (const image of imageFiles) {

        const key = image.name.split("_")[0].toLowerCase();

        const excel = excelMap[key];

        if (!excel) {
            console.warn("Excel mapping not found:", image.name);
            continue;
        }

        try {

            callbacks.onProcessing?.(image, completed, total);

            const png = await removeImageBackground(image);

            const result = await uploadProcessedImage(png, {
                originalName: image.name,
                key,
                category: excel.category,
                style: excel.style
            });

            completed++;

            callbacks.onSuccess?.(image, result, completed, total);

        } catch (err) {

            console.error(image.name, err);

            callbacks.onError?.(image, err, completed, total);
        }
    }

    callbacks.onComplete?.(completed, total);
}
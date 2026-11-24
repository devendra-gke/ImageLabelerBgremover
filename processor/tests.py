import io

import pandas as pd
from django.test import TestCase
from PIL import Image

from processor.services.image_processor import load_preset_settings, read_excel_mapping


class JewelryBackgroundRemovalTests(TestCase):
    def test_load_jewellery_preset_from_json(self):
        preset = load_preset_settings("jewellery")

        self.assertEqual(preset["preset_name"], "jewellery_image")
        self.assertTrue(preset["background_removal"]["preserve_prongs_and_claws"])
        self.assertTrue(preset["product_protection"]["no_detail_smoothing"])

    def test_excel_mapping_uses_column_n_for_names_and_e_to_k_for_text(self):
        cols = ["Finish Date", "Category", "Sub", "Metal", "StyleBio", "TagNo",
                "DiaWt", "ChainWt", "GrossWt", "StoneWt", "Remark", "OtherWt",
                "Description", "ImageName"]
        row = ["2025-01-01", "RING", "RING", "YELLOW GOLD", "SB-202", "7",
               "Di.Wt: 3 / 0.90 Ct", "", "Gr Wt: 12.50 Gm", "", "ok", "x",
               "desc", "My Photo 01.JPG"]
        buffer = io.BytesIO()
        pd.DataFrame([row], columns=cols).to_excel(buffer, index=False)
        buffer.seek(0)

        mapping, category_map, style_map = read_excel_mapping(buffer)

        self.assertEqual(
            mapping["my photo 01"],
            ["SB-202", "7", "Di.Wt: 3 / 0.90 Ct", "Gr Wt: 12.50 Gm", "ok"],
        )
        self.assertEqual(category_map["my photo 01"], "RING")
        self.assertEqual(style_map["my photo 01"], "SB-")


# ----------------------------------------------------------------------
# Jewellery detection / matte clean-up / 3D (no AI model needed: the
# segmentation call is replaced with a deterministic fake).
# ----------------------------------------------------------------------
from unittest import mock

import numpy as np
from django.core.files.uploadedfile import SimpleUploadedFile


def _ring_photo(w=900, h=700, cx=600, cy=380, r_out=120, r_in=95):
    """Gold ring on a vignetted light-grey backdrop, plus dust specks."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = 240 - 18 * np.hypot((xx - w / 2) / w, (yy - h / 2) / h)
    img = np.repeat(bg[..., None], 3, axis=2)
    dist = np.hypot(xx - cx, yy - cy)
    band = (dist <= r_out) & (dist >= r_in)
    img[band] = (205, 160, 70)
    img[100:102, 100:102] = 120          # dust speck far away
    return Image.fromarray(img.astype(np.uint8)), band, dist < r_in


def _fake_model(image, model_name):
    """A 'model' that fills the ring hole (a typical failure) at low precision."""
    arr = np.asarray(image.convert("RGB")).astype(np.int16)
    gold = (arr[..., 0] - arr[..., 2]) > 60
    from scipy.ndimage import binary_fill_holes
    filled = binary_fill_holes(gold)
    alpha = (filled * 255).astype(np.uint8)
    speck = (np.abs(arr - 120).max(axis=2) < 5)
    alpha[speck] = 255
    return alpha


class JewelleryDetectionTests(TestCase):
    def setUp(self):
        patcher = mock.patch(
            "processor.services.jewellery_detection.predict_alpha",
            side_effect=_fake_model,
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_detects_ring_box_and_clears_hole_and_speck(self):
        from processor.services.jewellery_detection import cutout_jewellery

        photo, band, hole = _ring_photo()
        result = cutout_jewellery(photo, {})
        x0, y0, x1, y1 = result.bbox
        self.assertTrue(470 <= x0 <= 490 and 710 <= x1 <= 730, result.bbox)
        self.assertTrue(250 <= y0 <= 270 and 490 <= y1 <= 510, result.bbox)
        self.assertTrue(result.refined)
        # the band stays solid, the hollow and the dust are removed
        self.assertGreater((result.alpha[band] > 200).mean(), 0.97)
        self.assertLess((result.alpha[hole] > 0).mean(), 0.02)
        self.assertEqual(result.alpha[100:102, 100:102].max(), 0)

    def test_no_jewellery_returns_empty_detection(self):
        from processor.services.jewellery_detection import cutout_jewellery

        blank = Image.new("RGB", (400, 300), (240, 240, 240))
        result = cutout_jewellery(blank, {})
        self.assertIsNone(result.bbox)
        self.assertTrue(result.warnings)

    def test_auto_crop_frames_product_with_padding(self):
        from processor.services.image_processor import resize_to_canvas

        rgba = Image.new("RGBA", (1000, 800), (0, 0, 0, 0))
        rgba.paste((200, 150, 60, 255), (100, 100, 200, 150))
        canvas = resize_to_canvas(rgba, "#FFFFFF", 1750, 1160, auto_crop=True, padding_percent=10, max_upscale=20)
        arr = np.asarray(canvas)
        ys, xs = np.nonzero(arr[..., 0] < 250)
        # product width fills 80% of the canvas width (10% padding each side)
        self.assertAlmostEqual((xs.max() - xs.min() + 1) / 1750, 0.8, delta=0.01)

    def test_make_3d_endpoint_returns_textures(self):
        photo, _, _ = _ring_photo()
        buffer = io.BytesIO()
        photo.save(buffer, format="PNG")
        upload = SimpleUploadedFile("ring.png", buffer.getvalue(), content_type="image/png")
        response = self.client.post("/make-3d/", {"image": upload, "depth_mode": "shape"})
        self.assertEqual(response.status_code, 200, response.content[:300])
        data = response.json()
        self.assertEqual(data["depth_mode"], "shape")
        import base64
        depth = Image.open(io.BytesIO(base64.b64decode(data["depth"])))
        self.assertEqual(depth.size, (data["width"], data["height"]))
        packed = np.asarray(depth).astype(np.uint32)
        height = (packed[..., 0] << 8) | packed[..., 1]
        self.assertGreater(height.max(), 30000)          # the band is raised
        cutout = Image.open(io.BytesIO(base64.b64decode(data["cutout"])))
        a = np.asarray(cutout.getchannel("A"))
        self.assertEqual(int(height[a == 0].max()), 0)   # backdrop stays flat

    def test_process_images_transparent_mode_reports_detection(self):
        photo, _, _ = _ring_photo()
        buffer = io.BytesIO()
        photo.save(buffer, format="PNG")
        upload = SimpleUploadedFile("ring_TEST-1.png", buffer.getvalue(), content_type="image/png")
        response = self.client.post("/process-images/", {"images": upload, "processing_mode": "transparent_only"})
        item = response.json()["data"][0]
        self.assertEqual(item["status"], "success", item)
        self.assertEqual(item["detection"]["parts"], 1)
        self.assertTrue(item["transparent_url"].endswith("/Transparent/ring_TEST-1.png"))
        self.assertTrue(item["detection_preview"])


class DiamondShineTests(TestCase):
    """Stones must come out fully opaque even when the model leaves them
    semi-transparent (they reflect the backdrop); a backdrop gap between two
    settings must still be cleared."""

    def _scene(self):
        rng = np.random.default_rng(1)
        h, w = 400, 600
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        img = np.zeros((h, w, 3), np.float32)
        img[:] = (182, 175, 192)                                   # lavender backdrop
        stones = [(250, 200), (350, 200)]
        bezel = np.zeros((h, w), bool)
        stone = np.zeros((h, w), bool)
        for cx, cy in stones:
            d = np.hypot(xx - cx, yy - cy)
            ring = (d <= 40) & (d > 30)
            bezel |= ring
            img[ring] = (215, 160, 140)                             # rose gold
            st = d <= 30
            stone |= st
            facets = 175 + 60 * np.sign(np.sin(np.arctan2(yy - cy, xx - cx) * 8)) + rng.normal(0, 12, (h, w))
            img[st] = np.stack([facets, facets - 4, facets + 10], -1)[st]
            img[(np.hypot(xx - cx + 8, yy - cy + 8) < 4)] = 255       # sparkle
        bar = (np.abs(yy - 200) < 5) & (xx > 280) & (xx < 320)    # metal bar joining the bezels
        img[bar] = (215, 160, 140)
        gap = (np.abs(xx - 300) < 6) & (np.abs(yy - 228) < 8)       # backdrop gap under the bar
        return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)), bezel, stone, bar, gap

    def test_stones_become_opaque_and_gap_clears(self):
        from processor.services import hole_cleanup

        photo, bezel, stone, bar, gap = self._scene()
        alpha = np.zeros(stone.shape, np.uint8)
        alpha[bezel | bar] = 255
        alpha[stone] = 120                       # model unsure about the diamonds
        # model also fills the region between the bezels (incl. the gap)
        between = np.zeros_like(stone)
        between[185:245, 285:315] = True
        alpha[between & ~stone & ~bezel] = 250
        cleaned, stats, _ = hole_cleanup.clean_matte(np.asarray(photo), alpha, {})
        self.assertGreater((cleaned[stone] >= 250).mean(), 0.97, stats)
        self.assertLess((cleaned[gap] > 0).mean(), 0.2, stats)


class OpenworkHollowTests(TestCase):
    """A backdrop hollow inside a white-gold / pave frame (neutral, no
    coloured metal) that the model filled in must still be cleared."""

    def test_white_gold_frame_hollow_is_cleared(self):
        from processor.services import hole_cleanup

        rng = np.random.default_rng(3)
        h, w = 500, 700
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        img = np.zeros((h, w, 3), np.float32)
        img[:] = (211, 210, 215)                                      # light grey backdrop
        img += rng.normal(0, 1.0, img.shape)                          # sensor noise
        d = np.hypot((xx - 350) / 1.3, yy - 250)
        frame = (d <= 150) & (d >= 110)                                # pave frame
        hollow = d < 104
        pave = 200 + 45 * np.sign(np.sin(xx / 3) * np.sin(yy / 3)) + rng.normal(0, 10, (h, w))
        img[frame] = np.stack([pave, pave, pave + 4], -1)[frame]        # neutral, sparkly
        alpha = np.zeros((h, w), np.uint8)
        alpha[d <= 150] = 255                                           # model filled the hollow
        photo = np.clip(img, 0, 255).astype(np.uint8)
        cleaned, stats, _ = hole_cleanup.clean_matte(photo, alpha, {})
        self.assertLess((cleaned[hollow] > 0).mean(), 0.05, stats)
        inner_frame = frame & (d >= 113) & (d <= 147)
        self.assertGreater((cleaned[inner_frame] >= 250).mean(), 0.95, stats)


class GlassFrameTests(TestCase):
    def _photo_with_lens(self):
        import cv2
        img = np.full((600, 1000, 3), 255, np.uint8)
        # stadium outline with dark edge lines, like a magnifier inset
        cv2.rectangle(img, (150, 200), (850, 400), (50, 50, 50), 3)
        for x in range(150, 851, 25):
            cv2.line(img, (x, 200), (x, 215), (50, 50, 50), 2)
            cv2.line(img, (x, 385), (x, 400), (50, 50, 50), 2)
        for y in range(200, 401, 25):
            cv2.line(img, (150, y), (165, y), (50, 50, 50), 2)
            cv2.line(img, (835, y), (850, y), (50, 50, 50), 2)
        return img

    def test_lens_rim_is_restored_and_inside_left_alone(self):
        from processor.services.glass_frame import restore_glass_frame
        src = self._photo_with_lens()
        rgba = np.zeros((600, 1000, 4), np.uint8)
        added = restore_glass_frame(rgba, src)
        self.assertGreater(added, 1000)
        self.assertGreater(rgba[205, 500, 3], 200)    # rim kept
        self.assertEqual(rgba[300, 500, 3], 0)        # lens interior untouched

    def test_photo_without_lens_is_untouched(self):
        from processor.services.glass_frame import restore_glass_frame
        src = np.full((600, 1000, 3), 240, np.uint8)
        src[300:310, 100:900] = (20, 20, 20)           # a single dark strip
        rgba = np.zeros((600, 1000, 4), np.uint8)
        self.assertEqual(restore_glass_frame(rgba, src), 0)

import io

import numpy as np
import pandas as pd
from django.test import TestCase
from PIL import Image

from processor.services.image_processor import (
    preserve_jewelry_highlights,
    load_preset_settings,
    read_excel_mapping,
)


class JewelryBackgroundRemovalTests(TestCase):
    def test_preserve_bright_diamond_like_highlights(self):
        original = Image.new("RGBA", (10, 10), (28, 28, 28, 255))
        original.putpixel((5, 5), (245, 245, 250, 255))
        processed = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
        processed.putpixel((5, 5), (0, 0, 0, 0))

        fixed = preserve_jewelry_highlights(original, processed)
        rgba = np.asarray(fixed)

        self.assertEqual(rgba[5, 5, 3], 255)
        self.assertEqual(tuple(rgba[5, 5, :3]), (245, 245, 250))

    def test_load_jewellery_preset_from_json(self):
        preset = load_preset_settings("jewellery")

        self.assertEqual(preset["preset_name"], "jewellery_image")
        self.assertTrue(preset["background_removal"]["preserve_prongs_and_claws"])
        self.assertTrue(preset["product_protection"]["no_detail_smoothing"])

    def test_background_remains_transparent_outside_product(self):
        original = Image.new("RGBA", (10, 10), (255, 255, 255, 255))
        for x in range(3, 7):
            for y in range(3, 7):
                original.putpixel((x, y), (80, 80, 80, 255))
        original.putpixel((5, 5), (245, 245, 250, 255))

        processed = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
        for x in range(3, 7):
            for y in range(3, 7):
                processed.putpixel((x, y), (0, 0, 0, 255))
        processed.putpixel((5, 5), (0, 0, 0, 0))

        fixed = preserve_jewelry_highlights(original, processed)

        self.assertEqual(fixed.getpixel((0, 0))[3], 0)
        self.assertEqual(fixed.getpixel((5, 5))[3], 255)
        self.assertEqual(tuple(fixed.getpixel((5, 5))[:3]), (245, 245, 250))

    def test_excel_mapping_uses_tagno_and_selected_fields(self):
        df = pd.DataFrame([
            {
                "Category": "RING",
                "StyleBio": "SB-202",
                "TagNo": "1001",
                "DiaWt": "0.90",
                "ChainWt": "0.10",
                "GrossWt": "12.50",
                "StoneWt": "1.20",
                "Remark": "ok",
                "Description": "gold ring",
            }
        ])

        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        buffer.seek(0)

        mapping, category_map, _ = read_excel_mapping(buffer)

        self.assertIn("1001", mapping)
        self.assertEqual(mapping["1001"], ["SB-202", "1001", "0.90", "0.10", "12.50", "1.20"])
        self.assertEqual(category_map["1001"], "RING")

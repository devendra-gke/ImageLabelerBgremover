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

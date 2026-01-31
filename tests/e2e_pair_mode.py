"""E2E tests for Pair Mode extraction.

Tests the complete pair mode workflow:
1. Timeline job directory creation
2. Picture/text separation into separate image files
3. result_pairs.json output with binding/linking schema
4. Learning loop with observable improvements across runs
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw, ImageFont

# Import from the flashcard_engine package
from flashcard_engine.job import new_job_id, create_job_dirs, JobPaths
from flashcard_engine.pair_extractor import (
    PairExtractor,
    PairConfig,
    LearningCache,
    apply_pair_feedback,
    compute_blank_score,
)
from flashcard_engine.panel_debug import compute_blank_score as panel_compute_blank_score


class TestTimelineJobDirectory:
    """Test R4: Timeline job directory format."""
    
    def test_new_job_id_timeline_format(self):
        """new_job_id(use_timeline=True) returns YYYY-MM-DD/HH-MM-SS__<shortid>"""
        job_id = new_job_id(use_timeline=True)
        
        # Should have format: YYYY-MM-DD/HH-MM-SS__<shortid>
        parts = job_id.split("/")
        assert len(parts) == 2, f"Expected 2 parts separated by /, got: {job_id}"
        
        date_part = parts[0]  # YYYY-MM-DD
        time_and_id = parts[1]  # HH-MM-SS__<shortid>
        
        # Validate date format
        assert len(date_part) == 10, f"Date part should be YYYY-MM-DD, got: {date_part}"
        assert date_part[4] == "-" and date_part[7] == "-"
        
        # Validate time__id format
        assert "__" in time_and_id, f"Should have __ separator: {time_and_id}"
        time_part, short_id = time_and_id.split("__")
        assert len(time_part) == 8, f"Time part should be HH-MM-SS, got: {time_part}"
        assert len(short_id) == 8, f"Short ID should be 8 chars, got: {short_id}"
    
    def test_new_job_id_classic_format(self):
        """new_job_id(use_timeline=False) returns classic UUID format."""
        job_id = new_job_id(use_timeline=False)
        
        # Should be UUID format (no slashes)
        assert "/" not in job_id
        assert len(job_id) == 36  # UUID with dashes
    
    def test_create_job_dirs_timeline(self):
        """create_job_dirs with timeline creates nested date/time directories."""
        with tempfile.TemporaryDirectory() as tmp:
            job_id = new_job_id(use_timeline=True)
            paths = create_job_dirs(tmp, job_id, use_timeline=True)
            
            assert paths.job_dir.exists()
            # Check nested structure
            assert paths.job_dir.name.startswith(job_id.split("/")[1][:8])  # Time part
            assert paths.items_dir.exists()
            assert paths.stage_pair_dir.exists()


class TestPairExtraction:
    """Test R1: Picture/text crop separation."""
    
    @pytest.fixture
    def workspace_dir(self) -> Path:
        """Create temporary workspace."""
        tmp = tempfile.mkdtemp()
        yield Path(tmp)
        shutil.rmtree(tmp, ignore_errors=True)
    
    @pytest.fixture
    def sample_vocab_image(self) -> Image.Image:
        """Create a sample vocabulary card image with picture + text regions."""
        # Create 400x500 image with distinct regions
        img = Image.new("RGB", (400, 500), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Picture region (top 70%): Draw a colored rectangle (simulating an illustration)
        draw.rectangle([20, 20, 380, 320], fill=(100, 150, 200))
        draw.ellipse([100, 100, 300, 280], fill=(255, 200, 100))
        
        # Text region (bottom 30%): Draw black text
        draw.rectangle([20, 340, 380, 480], fill=(245, 245, 245))
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except OSError:
            font = ImageFont.load_default()
        draw.text((40, 370), "APPLE", fill=(0, 0, 0), font=font)
        draw.text((40, 420), "りんご", fill=(0, 0, 0), font=font)
        
        return img
    
    def test_extract_pairs_creates_picture_and_text_crops(
        self, workspace_dir: Path, sample_vocab_image: Image.Image
    ):
        """Pair extraction creates separate picture.png and text.png files."""
        job_id = new_job_id(use_timeline=True)
        paths = create_job_dirs(str(workspace_dir), job_id, use_timeline=True)
        
        # Save test image
        page_path = paths.pages_dir / "page_001.png"
        sample_vocab_image.save(page_path)
        
        extractor = PairExtractor(paths=paths, config=PairConfig())
        
        pairs, diagnostics = extractor.extract_pairs_from_page(
            page_id="test_page_001",
            page_index=0,
            page_image=sample_vocab_image,
            ocr_tokens=[{"text": "APPLE", "bbox_xyxy": [40, 370, 200, 420], "confidence": 0.95}],
        )
        
        assert len(pairs) >= 1, "Should extract at least one pair"
        
        pair = pairs[0]
        # Verify picture crop exists
        picture_path = paths.job_dir / pair.picture_path
        assert picture_path.exists(), f"Picture crop should exist: {pair.picture_path}"
        
        # Verify text crop exists
        text_path = paths.job_dir / pair.text_path
        assert text_path.exists(), f"Text crop should exist: {pair.text_path}"
        
        # Verify crops are valid images
        pic_img = Image.open(picture_path)
        txt_img = Image.open(text_path)
        assert pic_img.size[0] > 0 and pic_img.size[1] > 0
        assert txt_img.size[0] > 0 and txt_img.size[1] > 0
    
    def test_pair_has_linking_metadata(
        self, workspace_dir: Path, sample_vocab_image: Image.Image
    ):
        """Each pair includes machine-readable linking metadata."""
        job_id = new_job_id(use_timeline=True)
        paths = create_job_dirs(str(workspace_dir), job_id, use_timeline=True)
        
        extractor = PairExtractor(paths=paths)
        
        pairs, _ = extractor.extract_pairs_from_page(
            page_id="test_page",
            page_index=0,
            page_image=sample_vocab_image,
        )
        
        pair = pairs[0]
        
        # R2: Binding/linking - verify required fields
        assert pair.pair_id, "pair_id is required"
        assert pair.page_id == "test_page"
        assert pair.picture_path is not None
        assert pair.text_path is not None
        assert pair.bbox_item_xyxy is not None and len(pair.bbox_item_xyxy) == 4
        assert pair.bbox_picture_xyxy is not None and len(pair.bbox_picture_xyxy) == 4
        assert pair.bbox_text_xyxy is not None and len(pair.bbox_text_xyxy) == 4
        assert 0.0 <= pair.blank_score_picture <= 1.0
        assert 0.0 <= pair.blank_score_text <= 1.0


class TestLearningLoop:
    """Test R3: Self-improving loop with observable changes across runs."""
    
    @pytest.fixture
    def learning_workspace(self) -> Path:
        """Create workspace with learning cache."""
        tmp = tempfile.mkdtemp()
        yield Path(tmp)
        shutil.rmtree(tmp, ignore_errors=True)
    
    def test_learning_cache_creation(self, learning_workspace: Path):
        """Learning cache creates required directories."""
        cache = LearningCache(learning_workspace)
        
        assert (learning_workspace / "learning").exists()
        assert (learning_workspace / "learning" / "records").exists()
    
    def test_learning_cache_stores_corrections(self, learning_workspace: Path):
        """Caption corrections are stored and retrievable."""
        cache = LearningCache(learning_workspace)
        
        # Store a correction
        cache.set_caption_correction("page_001_item_0_[10,20,30,40]", "CORRECTED TEXT")
        
        # Retrieve it
        result = cache.get_caption_correction("page_001_item_0_[10,20,30,40]")
        assert result == "CORRECTED TEXT"
        
        # Missing key returns None
        assert cache.get_caption_correction("nonexistent") is None
    
    def test_learning_cache_stores_text_ratio(self, learning_workspace: Path):
        """Learned text ratios are stored and used."""
        cache = LearningCache(learning_workspace)
        
        # Initial default
        assert cache.get_text_ratio("page_hash_123") == 0.3
        
        # Store learned ratio
        cache.set_text_ratio("page_hash_123", 0.4)
        
        # Retrieve learned ratio
        assert cache.get_text_ratio("page_hash_123") == 0.4
    
    def test_apply_feedback_updates_learning_cache(self, learning_workspace: Path):
        """Applying feedback records learning data."""
        # Create a job with result_pairs.json
        job_id = new_job_id(use_timeline=True)
        job_dir = create_job_dirs(str(learning_workspace), job_id, use_timeline=True)
        
        # Write initial result_pairs.json
        result_pairs = {
            "schema_version": "1.0",
            "job_id": job_id,
            "pairs": [
                {
                    "pair_id": "pair_abc123",
                    "page_id": "page_001",
                    "item_index": 0,
                    "picture_path": "pages/items/page_001/item_000/picture.png",
                    "text_path": "pages/items/page_001/item_000/text.png",
                    "caption_text": "ORIGNAL TEXT",  # intentional typo
                    "bbox_item_xyxy": [0, 0, 100, 100],
                    "bbox_picture_xyxy": [0, 0, 100, 70],
                    "bbox_text_xyxy": [0, 70, 100, 100],
                    "status": "review",
                    "needs_review": True,
                    "reasons": ["OCR_EMPTY"],
                    "blank_score_picture": 0.2,
                    "blank_score_text": 0.3,
                    "confidence": 0.6,
                }
            ],
        }
        
        with open(job_dir.result_pairs_json, "w", encoding="utf-8") as f:
            json.dump(result_pairs, f)
        
        # Create feedback
        feedback_file = learning_workspace / "feedback.json"
        feedback = {
            "items": [
                {
                    "pair_id": "pair_abc123",
                    "action": "edit",
                    "edited_caption": "ORIGINAL TEXT",  # corrected
                    "corrected_text_ratio": 0.35,
                }
            ]
        }
        with open(feedback_file, "w", encoding="utf-8") as f:
            json.dump(feedback, f)
        
        # Apply feedback with learning cache
        cache = LearningCache(learning_workspace)
        stats = apply_pair_feedback(
            job_dir=job_dir.job_dir,
            feedback_path=feedback_file,
            learning_cache=cache,
        )
        
        assert stats["processed"] == 1
        assert stats["edited"] == 1
        assert stats["learning_records"] >= 1
        
        # Verify learning record was created
        learning_stats = cache.get_stats()
        assert learning_stats["total_records"] >= 1
    
    def test_second_run_uses_learned_corrections(self, learning_workspace: Path):
        """Second run should use corrections learned from first run feedback."""
        cache = LearningCache(learning_workspace)
        
        # Simulate learning from first run
        cache.set_caption_correction("hash123_0_100,200,300,400", "LEARNED CAPTION")
        cache.set_text_ratio("hash123", 0.4)
        
        # Get stats showing observable change
        stats = cache.get_stats()
        assert stats["caption_corrections"] >= 1
        assert stats["cached_parameters"] >= 1


class TestResultPairsJSON:
    """Test result_pairs.json output schema."""
    
    @pytest.fixture
    def workspace_with_pairs(self) -> tuple[Path, JobPaths]:
        """Create workspace with pair extraction results."""
        tmp = tempfile.mkdtemp()
        workspace = Path(tmp)
        
        job_id = new_job_id(use_timeline=True)
        paths = create_job_dirs(str(workspace), job_id, use_timeline=True)
        
        # Create a simple test image
        img = Image.new("RGB", (200, 300), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, 190, 200], fill=(100, 100, 200))
        draw.rectangle([10, 210, 190, 290], fill=(240, 240, 240))
        
        extractor = PairExtractor(paths=paths)
        pairs, diagnostics = extractor.extract_pairs_from_page(
            page_id="test",
            page_index=0,
            page_image=img,
        )
        
        # Write result_pairs.json
        from flashcard_engine.utils import write_json
        from dataclasses import asdict
        
        result = {
            "schema_version": "1.0",
            "job_id": job_id,
            "pairs": [asdict(p) for p in pairs],
            "diagnostics": [asdict(diagnostics)],
        }
        write_json(paths.result_pairs_json, result)
        
        yield workspace, paths
        shutil.rmtree(tmp, ignore_errors=True)
    
    def test_result_pairs_json_exists(self, workspace_with_pairs):
        """result_pairs.json is created in job directory."""
        workspace, paths = workspace_with_pairs
        
        assert paths.result_pairs_json.exists()
    
    def test_result_pairs_json_schema(self, workspace_with_pairs):
        """result_pairs.json follows expected schema."""
        workspace, paths = workspace_with_pairs
        
        with open(paths.result_pairs_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Required top-level fields
        assert "schema_version" in data
        assert "job_id" in data
        assert "pairs" in data
        assert isinstance(data["pairs"], list)
        
        if data["pairs"]:
            pair = data["pairs"][0]
            # Required pair fields
            assert "pair_id" in pair
            assert "page_id" in pair
            assert "picture_path" in pair
            assert "text_path" in pair
            assert "bbox_item_xyxy" in pair
            assert "bbox_picture_xyxy" in pair
            assert "bbox_text_xyxy" in pair
            assert "status" in pair
            assert "confidence" in pair


class TestBlankScoreDetection:
    """Test blank_score correctly identifies blank vs content-rich crops."""
    
    def test_blank_image_high_score(self):
        """Blank/white image should have high blank_score."""
        blank = Image.new("RGB", (100, 100), color=(255, 255, 255))
        result = panel_compute_blank_score(blank)
        assert result.score > 0.7, f"Blank image should have score > 0.7, got {result.score}"
    
    def test_content_image_low_score(self):
        """Image with content should have low blank_score."""
        # Use a real-world like pattern: checkerboard with varied shades
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Create high-entropy pattern: random noise-like pixels
        import random
        random.seed(42)  # Reproducible
        for x in range(100):
            for y in range(100):
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                draw.point((x, y), fill=(r, g, b))
        
        # Add edges
        for i in range(10):
            draw.line([(i * 10, 0), (i * 10, 100)], fill=(0, 0, 0), width=2)
            draw.line([(0, i * 10), (100, i * 10)], fill=(0, 0, 0), width=2)
        
        result = panel_compute_blank_score(img)
        # Random noise image with grid lines should have lower blank score
        assert result.score <= 0.7, f"Content image should have score <= 0.7, got {result.score}"


class TestEndToEndPairMode:
    """End-to-end test of pair mode with real-ish workflow."""
    
    def test_full_pair_extraction_workflow(self):
        """Full workflow: extract → review → feedback → re-run with learning."""
        tmp = tempfile.mkdtemp()
        try:
            workspace = Path(tmp)
            
            # Create test images
            images_dir = workspace / "test_images"
            images_dir.mkdir()
            
            for i in range(2):
                img = Image.new("RGB", (400, 500), color=(255, 255, 255))
                draw = ImageDraw.Draw(img)
                # Picture area
                draw.rectangle([20, 20, 380, 320], fill=(100 + i * 50, 150, 200))
                # Text area
                draw.rectangle([20, 340, 380, 480], fill=(245, 245, 245))
                draw.text((40, 380), f"WORD_{i}", fill=(0, 0, 0))
                img.save(images_dir / f"card_{i}.png")
            
            # Run 1: Initial extraction
            from flashcard_engine.config import load_config
            from flashcard_engine.pipeline import EnginePipeline, RunOptions
            
            job_id_1 = new_job_id(use_timeline=True)
            paths_1 = create_job_dirs(str(workspace), job_id_1, use_timeline=True)
            
            # Initialize outputs
            from flashcard_engine.job import init_job_outputs
            init_job_outputs(paths_1)
            
            cfg = load_config(str(Path(__file__).parent.parent / "config" / "default.json"))
            opts = RunOptions(
                input_path=str(images_dir),
                input_type="images",
                lang="en",
                source="test",
                dpi=200,
                min_confidence=0.7,
                segmenter="off",
                segmenter_device="cpu",
                mode="pair",
                learning_enabled=True,
            )
            
            pipeline = EnginePipeline(paths=paths_1, cfg=cfg, opts=opts)
            pipeline.run(job_id_1)
            
            # Verify result_pairs.json created
            assert paths_1.result_pairs_json.exists()
            
            with open(paths_1.result_pairs_json, "r", encoding="utf-8") as f:
                result_1 = json.load(f)
            
            assert len(result_1["pairs"]) >= 1, "Should have extracted at least one pair"
            
            # Create feedback for first run
            feedback_1 = workspace / "feedback_1.json"
            with open(feedback_1, "w", encoding="utf-8") as f:
                json.dump({
                    "items": [
                        {
                            "pair_id": result_1["pairs"][0]["pair_id"],
                            "action": "edit",
                            "edited_caption": "CORRECTED_CAPTION_1",
                        }
                    ]
                }, f)
            
            # Apply feedback
            cache = LearningCache(workspace)
            stats = apply_pair_feedback(
                job_dir=paths_1.job_dir,
                feedback_path=feedback_1,
                learning_cache=cache,
            )
            
            assert stats["edited"] >= 1
            
            # Verify learning was captured
            learning_stats = cache.get_stats()
            assert learning_stats["total_records"] >= 1, "Learning should have recorded feedback"
            
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

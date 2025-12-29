#!/usr/bin/env python3
"""
SMART Vision Agent - Automatically skips text-only images
Only analyzes visual maps, ignoring text screenshots
"""

import os
import json
import base64
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import logging

from openai import OpenAI
import re

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

@dataclass
class VisionConfig:
    """Configuration for vision analysis"""
    model: str = "gpt-4o"
    max_plans_to_analyze: int = 5
    initial_detail: str = "low"
    final_detail: str = "high"
    initial_max_tokens: int = 300
    final_max_tokens: int = 1500
    parallel_workers: int = 1          # Sequential for slow connections
    timeout_per_plan: int = 30         # Longer timeout
    total_timeout: int = 90            # Longer total timeout
    cache_results: bool = True
    auto_skip_text_images: bool = True  # NEW: Auto-skip text-only images
    

class SmartVisionAgent:
    """
    Enhanced Vision Agent that automatically detects and skips text-only images.
    Only analyzes visual maps/plans, saving time and money.
    """
    
    def __init__(
        self, 
        openai_api_key: str, 
        plans_directory: str = "data/raw/Landuse Plans",
        config: Optional[VisionConfig] = None
    ):
        self.client = OpenAI(api_key=openai_api_key)
        self.plans_dir = Path(plans_directory)
        self.config = config or VisionConfig()
        
        # Caches
        self._analysis_cache = {}
        self._image_type_cache = {}  # NEW: Cache image type (visual/text)
        
        # Metrics
        self.metrics = {
            'total_calls': 0,
            'cache_hits': 0,
            'timeouts': 0,
            'errors': 0,
            'total_tokens_used': 0,
            'total_time': 0.0,
            'text_images_skipped': 0  # NEW: Count skipped text images
        }
        
        logger.info(f"🎨 Smart Vision Agent initialized")
        logger.info(f"   Auto-skip text images: {self.config.auto_skip_text_images}")
    
    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _is_visual_map(self, image_path: Path) -> bool:
        """
        Quick check: Is this image a VISUAL MAP or just TEXT?
        Uses low-detail, fast check. Results are cached.
        
        Returns:
            True if visual map (should analyze)
            False if text-only (should skip)
        """
        # Check cache first
        cache_key = f"type_{image_path.stem}_{image_path.stat().st_mtime}"
        if cache_key in self._image_type_cache:
            logger.info(f"📦 Using cached image type for {image_path.name}")
            return self._image_type_cache[cache_key]
        
        try:
            logger.info(f"🔍 Quick check: {image_path.name}")
            
            image_data = self._encode_image(image_path)
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Is this a VISUAL MAP/PLAN or just TEXT?

Answer ONLY with ONE WORD:
- "VISUAL" if it shows a map, site plan, cadastral plan, or drawn plot boundaries
- "TEXT" if it's primarily paragraphs of text, tables, or cover pages
- "MIXED" if it has both map and significant text

One word answer:"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}",
                                    "detail": "low"  # Fast, cheap check
                                }
                            }
                        ]
                    }
                ],
                max_tokens=10,
                temperature=0
            )
            
            result = response.choices[0].message.content.strip().upper()
            
            # VISUAL or MIXED = analyze
            # TEXT = skip
            is_visual = "VISUAL" in result or "MIXED" in result
            
            if is_visual:
                logger.info(f"   ✅ {result} - Will analyze")
            else:
                logger.info(f"   ⏭️  {result} - Skipping (text-only)")
                self.metrics['text_images_skipped'] += 1
            
            # Cache result
            self._image_type_cache[cache_key] = is_visual
            
            return is_visual
            
        except Exception as e:
            logger.error(f"Error checking image type: {e}")
            # On error, assume it's visual (don't skip)
            return True
    
    def _get_available_plans(self) -> List[Path]:
        """Get all available plan images"""
        if not self.plans_dir.exists():
            logger.warning(f"Plans directory not found: {self.plans_dir}")
            return []
        
        plans = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
            plans.extend(self.plans_dir.rglob(f"images/{ext}"))
            plans.extend(self.plans_dir.rglob(ext))
        
        plans = list(set(plans))
        logger.info(f"📐 Found {len(plans)} plan images")
        return plans
    
    def _filter_relevant_plans(self, query: str, all_plans: List[Path]) -> List[Path]:
        """Filter to relevant plans based on keywords"""
        if not all_plans:
            return []
        
        keywords = self._extract_location_keywords(query)
        
        if not keywords:
            logger.warning("⚠️ No location keywords found")
            return all_plans[:self.config.max_plans_to_analyze]
        
        # Score all plans
        scored_plans = [
            (plan, self._score_plan_relevance(plan, keywords))
            for plan in all_plans
        ]
        
        scored_plans.sort(key=lambda x: x[1], reverse=True)
        relevant = [plan for plan, score in scored_plans if score > 0.0]
        
        if not relevant:
            logger.warning(f"⚠️ No plans matched keywords {keywords}, using first plans")
            return all_plans[:self.config.max_plans_to_analyze]
        
        relevant = relevant[:self.config.max_plans_to_analyze]
        
        logger.info(f"📊 Filtered {len(all_plans)} plans → {len(relevant)} relevant plans")
        logger.info(f"   Keywords: {keywords}")
        
        return relevant
    
    def _extract_location_keywords(self, query: str) -> List[str]:
        """Extract keywords from query"""
        keywords = []
        query_lower = query.lower()
        
        # Stuttgart districts
        districts = {
            'nord': ['nord', 'north'],
            'süd': ['süd', 'sued', 'south'],
            'nordbahnhof': ['nordbahnhof', 'nordbahnhofstrasse']
        }
        
        for district, variants in districts.items():
            if any(v in query_lower for v in variants):
                keywords.extend(variants)
        
        # Extract plot numbers
        plot_matches = re.findall(r'(\d{3,5})[/-]?\d*', query)
        keywords.extend(plot_matches)
        
        return list(set(keywords))
    
    def _score_plan_relevance(self, plan_path: Path, keywords: List[str]) -> float:
        """Score plan relevance to query"""
        if not keywords:
            return 0.5
        
        filename = plan_path.name.lower()
        score = 0.0
        
        for keyword in keywords:
            if keyword in filename:
                score += 1.0
        
        return min(score / len(keywords), 1.0) if score > 0 else 0.0
    
    def find_plot_smart(self, plot_number: str) -> Dict[str, Any]:
        """
        Smart plot search that automatically skips text-only images
        """
        start_time = time.time()
        logger.info(f"🔍 Smart search for plot {plot_number}")
        
        # Get and filter plans
        all_plans = self._get_available_plans()
        relevant_plans = self._filter_relevant_plans(plot_number, all_plans)
        
        if not relevant_plans:
            return {
                'found': False,
                'error': 'No plan images available',
                'searched_directory': str(self.plans_dir)
            }
        
        # NEW: Filter out text-only images
        visual_plans = []
        
        if self.config.auto_skip_text_images:
            logger.info("🔍 Pre-filtering: checking which images are visual maps...")
            
            for plan in relevant_plans:
                if self._is_visual_map(plan):
                    visual_plans.append(plan)
            
            if not visual_plans:
                logger.warning("⚠️ All filtered plans are text-only!")
                # Fallback: try ALL plans (maybe filtering was wrong)
                logger.info("   Trying all plans as fallback...")
                visual_plans = all_plans[:self.config.max_plans_to_analyze]
            else:
                logger.info(f"✅ {len(visual_plans)} visual maps identified (skipped {len(relevant_plans) - len(visual_plans)} text images)")
        else:
            visual_plans = relevant_plans
        
        # Now analyze only the visual plans
        for plan in visual_plans:
            try:
                logger.info(f"📊 Analyzing: {plan.name}")
                
                analysis = self._analyze_plan_full(
                    plan,
                    f"Does this visual map show plot {plot_number} (Flurstück {plot_number})? If yes, describe its location and boundaries."
                )
                
                # Check if found
                if plot_number in analysis or "yes" in analysis.lower()[:200]:
                    # Found! Get detailed analysis
                    logger.info(f"✅ Found plot {plot_number} in {plan.name}")
                    
                    detailed = self._analyze_plan_full(
                        plan,
                        f"Provide detailed analysis of plot {plot_number}: zoning, boundaries, dimensions, setbacks.",
                        detail="high"
                    )
                    
                    elapsed = time.time() - start_time
                    
                    return {
                        'found': True,
                        'plan_file': plan.name,
                        'plan_path': str(plan),
                        'analysis': detailed,
                        'search_time': elapsed,
                        'plans_analyzed': len(visual_plans),
                        'text_images_skipped': self.metrics['text_images_skipped']
                    }
                    
            except Exception as e:
                logger.error(f"Error analyzing {plan.name}: {e}")
                continue
        
        # Not found
        elapsed = time.time() - start_time
        logger.warning(f"❌ Plot {plot_number} not found")
        
        return {
            'found': False,
            'searched_plans': len(visual_plans),
            'search_time': elapsed,
            'text_images_skipped': self.metrics['text_images_skipped'],
            'message': f'Plot {plot_number} not found in {len(visual_plans)} visual maps'
        }
    
    def _analyze_plan_full(self, plan_path: Path, query: str, detail: str = "low") -> str:
        """Analyze a plan with full Vision API call"""
        try:
            image_data = self._encode_image(plan_path)
            
            max_tokens = (self.config.initial_max_tokens if detail == "low" 
                         else self.config.final_max_tokens)
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""You are an expert at reading German site plans and cadastral maps.

{query}

Provide precise information. If you cannot find specific information, say so clearly."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}",
                                    "detail": detail
                                }
                            }
                        ]
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.1,
                timeout=self.config.timeout_per_plan
            )
            
            self.metrics['total_calls'] += 1
            if hasattr(response, 'usage'):
                self.metrics['total_tokens_used'] += response.usage.total_tokens
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.metrics['errors'] += 1
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.copy()
    
    def print_metrics(self):
        """Print metrics"""
        m = self.metrics
        print("\n" + "="*60)
        print("📊 SMART VISION AGENT METRICS")
        print("="*60)
        print(f"Total API calls:          {m['total_calls']}")
        print(f"Text images skipped:      {m['text_images_skipped']} (saved time/cost!)")
        print(f"Timeouts:                 {m['timeouts']}")
        print(f"Errors:                   {m['errors']}")
        print(f"Total tokens used:        {m['total_tokens_used']:,}")
        print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        sys.exit(1)
    
    config = VisionConfig(
        max_plans_to_analyze=5,
        parallel_workers=1,
        timeout_per_plan=30,
        total_timeout=90,
        auto_skip_text_images=True  # Enable smart filtering
    )
    
    agent = SmartVisionAgent(api_key, config=config)
    
    test_plot = "9232/79"
    print(f"\n🧪 Testing smart plot search: {test_plot}")
    print("="*60)
    print("⚡ Smart mode: Will auto-skip text-only images")
    print()
    
    result = agent.find_plot_smart(test_plot)
    
    print("\n" + "="*60)
    print("📊 RESULTS")
    print("="*60)
    
    if result['found']:
        print(f"✅ Found plot {test_plot}!")
        print(f"\n📁 Plan: {result['plan_file']}")
        print(f"⏱️  Time: {result['search_time']:.1f}s")
        print(f"📊 Plans analyzed: {result['plans_analyzed']}")
        print(f"⏭️  Text images skipped: {result['text_images_skipped']}")
        print(f"\n📄 Analysis:\n{result['analysis'][:500]}...")
    else:
        print(f"❌ Plot {test_plot} not found")
        print(f"📊 Searched: {result.get('searched_plans', 0)} visual maps")
        print(f"⏭️  Skipped: {result.get('text_images_skipped', 0)} text images")
        print(f"⏱️  Time: {result.get('search_time', 0):.1f}s")
    
    agent.print_metrics()
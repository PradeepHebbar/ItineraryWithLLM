#!/usr/bin/env python3
"""
Script to check available OpenAI models for your API key
"""
import os
from openai import OpenAI

def check_openai_models():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    try:
        client = OpenAI(api_key=api_key)
        models = client.models.list()
        
        print("Available OpenAI Models:")
        print("=" * 50)
        
        # Filter and categorize models
        gpt_models = []
        other_models = []
        
        for model in models.data:
            model_id = model.id
            if "gpt" in model_id.lower():
                gpt_models.append(model_id)
            else:
                other_models.append(model_id)
        
        print("\n🤖 GPT Models (recommended for chat):")
        for model in sorted(gpt_models):
            print(f"  - {model}")
        
        print(f"\n📊 Other Models ({len(other_models)} total):")
        for model in sorted(other_models)[:10]:  # Show first 10
            print(f"  - {model}")
        if len(other_models) > 10:
            print(f"  ... and {len(other_models) - 10} more")
        
        print(f"\n✅ Total models available: {len(models.data)}")
        
        # Recommend models for the travel planner
        recommended = []
        for model in gpt_models:
            if any(x in model for x in ["gpt-3.5", "gpt-4o", "gpt-4-turbo"]):
                recommended.append(model)
        
        if recommended:
            print("\n🎯 Recommended models for travel planning:")
            for model in recommended:
                print(f"  - {model}")
        
    except Exception as e:
        print(f"Error checking models: {e}")

if __name__ == "__main__":
    check_openai_models()

"""
Runner script for NLP to Trading Strategy Engine
Demonstrates parsing natural language trading rules into structured JSON.
"""
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from nlp_parser import NLParser, parse_trading_rule, check_completeness
from nlp_parser.schemas import CompletenessResponse, ParsedStrategy


def print_separator():
    """Print a visual separator."""
    print("\n" + "=" * 80 + "\n")


def display_parsed_result(result: ParsedStrategy):
    """Display parsed trading rule in a readable format."""
    print("📊 PARSED STRATEGY:")
    print(f"   Original Text: {result.raw_text}")
    print(f"   Confidence: {result.confidence:.2f}")
    
    if result.warnings:
        print("   ⚠️  Warnings:")
        for warning in result.warnings:
            print(f"      - {warning}")
    
    print("\n   📋 Rule Structure:")
    rule_dict = result.rule.model_dump(exclude_none=True)
    print(json.dumps(rule_dict, indent=4))


def example_1_simple_rule():
    """Example 1: Simple trading rule."""
    print_separator()
    print("🔹 EXAMPLE 1: Simple Trading Rule")
    print_separator()
    
    rule_text = "Buy when close price is above 20-day moving average"
    
    print(f"Input: '{rule_text}'")
    print("\nParsing...")
    
    try:
        result = parse_trading_rule(rule_text)
        display_parsed_result(result)
    except Exception as e:
        print(f"❌ Error: {e}")


def example_2_complex_rule():
    """Example 2: Complex rule with multiple conditions."""
    print_separator()
    print("🔹 EXAMPLE 2: Complex Rule with Multiple Conditions")
    print_separator()
    
    rule_text = "Buy when close > SMA(20) and volume > 1000000. Sell when RSI(14) < 30"
    
    print(f"Input: '{rule_text}'")
    print("\nParsing...")
    
    try:
        result = parse_trading_rule(rule_text)
        display_parsed_result(result)
    except Exception as e:
        print(f"❌ Error: {e}")


def example_3_incomplete_rule():
    """Example 3: Handling incomplete rule."""
    print_separator()
    print("🔹 EXAMPLE 3: Incomplete Rule Detection")
    print_separator()
    
    rule_text = "Use moving average strategy"
    
    print(f"Input: '{rule_text}'")
    print("\nChecking completeness...")
    
    parser = NLParser()
    completeness = parser.check_completeness(rule_text)
    
    print(f"\n📊 Completeness Check:")
    print(f"   Is Complete: {completeness.is_complete}")
    print(f"   Status: {completeness.status}")
    
    if not completeness.is_complete:
        print(f"   Message: {completeness.message}")
        if completeness.suggestions:
            print(f"   Suggestions:")
            for suggestion in completeness.suggestions:
                print(f"      - {suggestion}")


def example_4_crosses_above():
    """Example 4: Crosses above condition."""
    print_separator()
    print("🔹 EXAMPLE 4: Crosses Above Condition")
    print_separator()
    
    rule_text = "Enter when price crosses above yesterday's high"
    
    print(f"Input: '{rule_text}'")
    print("\nParsing...")
    
    try:
        result = parse_trading_rule(rule_text)
        display_parsed_result(result)
    except Exception as e:
        print(f"❌ Error: {e}")


def example_5_percentage_stop():
    """Example 5: Percentage-based stop loss."""
    print_separator()
    print("🔹 EXAMPLE 5: Percentage Stop Loss")
    print_separator()
    
    rule_text = "Sell when price drops 5% from entry"
    
    print(f"Input: '{rule_text}'")
    print("\nParsing...")
    
    try:
        result = parse_trading_rule(rule_text)
        display_parsed_result(result)
    except Exception as e:
        print(f"❌ Error: {e}")


def interactive_mode():
    """Interactive mode for testing custom rules."""
    print_separator()
    print("🎯 INTERACTIVE MODE")
    print("Enter your trading rules to parse them into structured JSON.")
    print("Type 'exit' to quit.")
    print_separator()
    
    parser = NLParser()
    
    while True:
        print("\n" + "-" * 80)
        rule_text = input("\n📝 Enter trading rule: ").strip()
        
        if rule_text.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not rule_text:
            continue
        
        print("\n1️⃣  Checking completeness...")
        try:
            completeness = parser.check_completeness(rule_text)
            
            print(f"   ✓ Is Complete: {completeness.is_complete}")
            print(f"   ✓ Status: {completeness.status}")
            
            if not completeness.is_complete:
                print(f"   ℹ️  Message: {completeness.message}")
                if completeness.suggestions:
                    print(f"   💡 Suggestions:")
                    for suggestion in completeness.suggestions:
                        print(f"      - {suggestion}")
                continue
            
            print("\n2️⃣  Parsing rule...")
            result = parser.parse(rule_text)
            display_parsed_result(result)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")


def run_all_examples():
    """Run all predefined examples."""
    print("\n" + "🚀" * 40)
    print("NLP TO TRADING STRATEGY ENGINE - DEMO")
    print("🚀" * 40)
    
    example_1_simple_rule()
    example_2_complex_rule()
    example_3_incomplete_rule()
    example_4_crosses_above()
    example_5_percentage_stop()
    
    print_separator()
    print("✅ All examples completed!")
    print_separator()


def main():
    """Main function with menu."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="NLP to Trading Strategy Engine Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_parser.py --examples        Run all predefined examples
  python run_parser.py --interactive     Start interactive mode
  python run_parser.py --parse "Buy when close > SMA(20)"
        """
    )
    
    parser.add_argument(
        '--examples',
        action='store_true',
        help='Run all predefined examples'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Start interactive mode'
    )
    
    parser.add_argument(
        '--parse',
        type=str,
        metavar='RULE',
        help='Parse a single trading rule'
    )
    
    parser.add_argument(
        '--check',
        type=str,
        metavar='RULE',
        help='Check completeness of a trading rule'
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        print("\n" + "💡" * 40)
        print("Tip: Start with --examples to see what this tool can do!")
        print("💡" * 40 + "\n")
        return
    
    # Run examples
    if args.examples:
        run_all_examples()
    
    # Parse single rule
    if args.parse:
        print_separator()
        print("🔍 PARSING SINGLE RULE")
        print_separator()
        print(f"Input: '{args.parse}'")
        print("\nParsing...")
        
        try:
            result = parse_trading_rule(args.parse)
            display_parsed_result(result)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Check completeness
    if args.check:
        print_separator()
        print("🔍 CHECKING COMPLETENESS")
        print_separator()
        print(f"Input: '{args.check}'")
        print("\nChecking...")
        
        parser_obj = NLParser()
        completeness = parser_obj.check_completeness(args.check)
        
        print(f"\n📊 Result:")
        print(f"   Is Complete: {completeness.is_complete}")
        print(f"   Status: {completeness.status}")
        
        if completeness.message:
            print(f"   Message: {completeness.message}")
        
        if completeness.suggestions:
            print(f"   Suggestions:")
            for suggestion in completeness.suggestions:
                print(f"      - {suggestion}")
    
    # Interactive mode
    if args.interactive:
        interactive_mode()


if __name__ == "__main__":
    main()

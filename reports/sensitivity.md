# Sensitivity (ASSUMED prices and costs; margin 60%)

Test set: 2,310 leads. 'vs call everyone' = model profit minus calling all 2,310 leads. 'vs random' = model profit minus calling the same number of leads in random order.

## Unconstrained (call while EV > 0)

| Price (assumed) | Cost/call (assumed) | Break-even P | % leads called | % enrolments captured | Profit: model (lakh) | vs call everyone (lakh) | vs random, same calls (lakh) |
|---|---|---|---|---|---|---|---|
| ₹5,000 | ₹50 | 1.7% | 96% | 99% | 25.4 | -0.1 | +0.9 |
| ₹5,000 | ₹150 | 5.0% | 87% | 98% | 23.2 | -0.0 | +2.9 |
| ₹5,000 | ₹300 | 10.0% | 73% | 96% | 20.4 | +0.6 | +5.9 |
| ₹15,000 | ₹50 | 0.6% | 98% | 100% | 78.8 | -0.2 | +1.5 |
| ₹15,000 | ₹150 | 1.7% | 96% | 99% | 76.3 | -0.3 | +2.6 |
| ₹15,000 | ₹300 | 3.3% | 92% | 99% | 72.8 | -0.4 | +5.3 |
| ₹30,000 | ₹50 | 0.3% | 100% | 100% | 159.0 | +0.0 | +0.0 |
| ₹30,000 | ₹150 | 0.8% | 97% | 100% | 156.1 | -0.6 | +3.6 |
| ₹30,000 | ₹300 | 1.7% | 96% | 99% | 152.6 | -0.6 | +5.1 |

## Capacity-limited (1,000 calls: 5 counsellors × 40 calls/day × 5 days)

| Price (assumed) | Cost/call (assumed) | Break-even P | % leads called | % enrolments captured | Profit: model (lakh) | vs call everyone (lakh) | vs random, same calls (lakh) |
|---|---|---|---|---|---|---|---|
| ₹5,000 | ₹50 | 1.7% | 43% | 77% | 20.2 | -5.4 | +9.1 |
| ₹5,000 | ₹150 | 5.0% | 43% | 77% | 19.2 | -4.1 | +9.1 |
| ₹5,000 | ₹300 | 10.0% | 43% | 77% | 17.7 | -2.1 | +9.1 |
| ₹15,000 | ₹50 | 0.6% | 43% | 77% | 61.5 | -17.4 | +27.3 |
| ₹15,000 | ₹150 | 1.7% | 43% | 77% | 60.5 | -16.1 | +27.3 |
| ₹15,000 | ₹300 | 3.3% | 43% | 77% | 59.0 | -14.2 | +27.3 |
| ₹30,000 | ₹50 | 0.3% | 43% | 77% | 123.5 | -35.5 | +54.7 |
| ₹30,000 | ₹150 | 0.8% | 43% | 77% | 122.5 | -34.2 | +54.7 |
| ₹30,000 | ₹300 | 1.7% | 43% | 77% | 121.0 | -32.2 | +54.7 |

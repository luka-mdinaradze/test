# სწრაფი დაწყება — $100 ტრენდ ბოტი

## 1. დაყენება

საჭიროა Python 3.11+.

```bash
cd trading-bot
pip install -r requirements.txt
```

## 2. ჯერ ტესტები (ყოველთვის)

```bash
python3 tests/test_engine.py    # 7 ტესტი — ბექტესტის ძრავი
python3 tests/test_bot.py       # 22 ტესტი — ბოტის რისკი, სტეიტი, ინვარიანტები
```

თუ რომელიმე FAIL-ს აჩვენებს — **არ გააგრძელო**. მითხარი რა წერია.

## 3. ბექტესტი რეალურ მონაცემებზე

```bash
python3 -m bot.run backtest
```

ავტომატურად ჩამოტვირთავს Binance-დან BTCUSDT 1h (2017 წლიდან, ~70,000 ბარი).
თუ Binance დაბლოკილია შენს ქსელში — გამოიყენე შენი CSV:

```bash
python3 run_all.py --csv შენი_ფაილი.csv
```

CSV-ს სჭირდება სვეტები: `open_time,open,high,low,close,volume`

**რას უყურებ:** `Sharpe`, `max_drawdown_pct`, `expectancy_R`, `trades`.
თუ `expectancy_R` უარყოფითია — სტრატეგიას ამ მონაცემებზე edge არ აქვს. ნუ გაუშვებ.

## 4. სრული კვლევა (walk-forward — ყველაზე მნიშვნელოვანი)

```bash
python3 run_all.py
```

მე-6 სექციაში ნახავ `Overfitting tax`-ს. **ეს არის მთავარი რიცხვი.**
თუ in-sample Sharpe 1.8-ია და out-of-sample 0.2 — სტრატეგია მორგებულია,
რეალურად არ მუშაობს. თუ სხვაობა 0.3-ზე ნაკლებია — edge სავარაუდოდ ნამდვილია.

## 5. Paper — ფორვარდ ტესტი

```bash
python3 -m bot.run paper          # რეალური ფასები, სიმულირებული შესრულება
python3 -m bot.run status         # სად დგახარ
```

ბოტი იწყებს PAPER ფაზით. აგროვებს გარიგებებს `bot_state.json`-ში.

## 6. LIVE-ზე გადასვლა

```bash
python3 -m bot.run promote
```

**უარს იტყვის**, თუ არ გაქვს 40+ paper გარიგება დადებითი expectancy-ით.
ეს განზრახაა — ბოტი თავად ვერ გადაწყვეტს, რომ მზადაა.

შემდეგ **აუცილებლად testnet-ზე ჯერ**:

```bash
export BINANCE_API_KEY=შენი_testnet_key
export BINANCE_API_SECRET=შენი_testnet_secret
python3 -m bot.run live --i-understand-the-risk --base https://testnet.binance.vision
```

testnet key უფასოა: https://testnet.binance.vision

მხოლოდ მას შემდეგ, რაც testnet-ზე ყველაფერი მუშაობს, გადადი რეალურზე
(`--base https://api.binance.com`).

## 7. გაჩერება

```bash
Ctrl+C                            # სტეიტი ინახება, სტოპი ბირჟაზე რჩება
python3 -m bot.run reset-halt     # halt-ის გასუფთავება (ადამიანის გადაწყვეტილება)
```

---

## მთავარი პარამეტრები — `bot/config.py`

| პარამეტრი | ნაგულისხმევი | რას ნიშნავს |
|---|---|---|
| `risk_pct` | 0.01 | რისკი გარიგებაზე (1%). **ნუ გაზრდი 0.02-ზე მეტად.** |
| `max_drawdown_pct` | 0.20 | ამაზე მეტი ჩავარდნა → ბოტი ჩერდება |
| `max_consecutive_losses` | 5 | ზედიზედ წაგებები → ჩერდება |
| `daily_loss_limit_R` | 3.0 | დღეში -3R → პაუზა მეორე დღემდე |
| `maker_only` | True | **არ გამორთო.** სამჯერ იაფია. |
| `min_paper_trades` | 40 | LIVE-ზე გადასვლის ბარიერი |

## უსაფრთხოების თვისებები

- ღია პოზიციას **ყოველთვის** აქვს სტოპ-ორდერი **ბირჟაზე** — თუ ბოტი მოკვდება, ანგარიში დაცულია
- სტეიტი ატომურად იწერება — კრახი ვერ დატოვებს ნახევრად დაწერილ ფაილს
- გადატვირთვისას ბირჟა არის ჭეშმარიტების წყარო, არა ლოკალური ფაილი
- `client_order_id` ყოველ ორდერზე — timeout-ის შემდეგ ხელახალი გაგზავნა ვერ შექმნის დუბლს
- spot, ბერკეტის გარეშე — ვიწრო სტოპზე ზომა მცირდება, არ ისესხება

## თუ რამე გატყდა

`bot.log`-ში ყველაფერი წერია. გამომიგზავნე ბოლო 50 ხაზი:

```bash
tail -50 bot.log
```

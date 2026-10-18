# tradebot — ბოტის პლატფორმა (შაბლონი)

**პლატფორმა** ყველაფერს აკეთებს, რაც ყველა სტრატეგიისთვის ერთნაირია:
- მონაცემების მიღება;
- ორდერების გაგზავნა;
- პოზიციის ზომის გამოთვლა;
- რისკის ლიმიტები;
- სესიის წესები (როდის შეიძლება შესვლა, როდის უნდა დაიხუროს ყველაფერი);
- მდგომარეობის შენახვა და ჟურნალი.

**სტრატეგია** ერთი პატარა კლასია ერთი მეთოდით, `on_bar`. ის უყურებს დახურულ ბარებს და პასუხობს: LONG, SHORT, EXIT ან არაფერი. ამ ეტაპზე სტრატეგია ცარიელი შაბლონია (`tradebot/strategies/template.py`) და **არასოდეს ვაჭრობს**, სანამ წესებს არ ჩავწერთ.

Backtest, paper და live ერთსა და იმავე ციკლში გადის. რასაც ტესტავ, ზუსტად ის მუშაობს.

```
სტრატეგია (შენი წესები)  →  Signal(long/short, stop, target)
                                   ↓
პლატფორმა:  რისკის შემოწმება → სესიის შემოწმება → ზომა → ორდერი stop-ით → ჟურნალი
                                   ↓
ბროკერი:    paper (სიმულაცია)  |  ccxt (კრიპტო ბირჟები)  |  IBKR (MNQ / NQ ფიუჩერსები)
```

---

## 1. MacBook-ზე გახსნა

### ყველაზე მარტივი გზა: მართვის პანელი (ერთი ფანჯარა)

საჭიროა Python 3.11 ან უფრო ახალი. შემოწმება: `python3 --version`. თუ ძველია: `brew install python@3.12`.

1. ჩამოტვირთე პროექტი: GitHub Desktop-ით ან ქვემოთ მოცემული `git clone`-ით. ბრენჩი: `claude/optimistic-curie-o1ogja`.
2. Finder-ში გახსენი საქაღალდე და **ორჯერ დააწკაპუნე `start_dashboard.command`-ზე**.
   - პირველ ჯერზე ერთხელ დააყენებს ყველაფერს (1–2 წუთი). მერე ბრაუზერში გაიხსნება http://localhost:8501.
   - თუ Mac-მა თქვა „ვერ გაიხსნება, დეველოპერი უცნობია": მარჯვენა ღილაკი → **Open** → **Open**. ეს ერთხელაა საჭირო.
   - თუ თქვა „permission denied": Terminal-ში ერთხელ გაუშვი `chmod +x start_dashboard.command`.
3. პანელის დასახურად დახურე Terminal-ის ის ფანჯარა, რომელიც გაიხსნა. **ბოტი ამით არ ჩერდება**: ის ცალკე პროცესია და მხოლოდ ღილაკით ჩერდება.

ან Terminal-იდან, დაყენების შემდეგ: `python3 -m tradebot dashboard`.

| ჩანართი | რა არის |
|---|---|
| 📊 Backtest | აირჩიე სტრატეგია, მონაცემი და რისკი, მერე დააჭირე ▶. ჩანს ბალანსის გრაფიკი, ფასი გარიგებებით (მაუსით ნახავ თითოეულს), ცხრილი და `trades.csv`. შეგიძლია შენი CSV ატვირთო. |
| 🤖 ბოტი | მარცხნივ აირჩიე paper ან live config. ღილაკები: ▶ ჩართვა · ⏹ გაჩერება (kill switch: ღია პოზიცია დაიხურება, მერე ბოტი ჩერდება) · ⛔ ახლავე (Ctrl+C) · ↺ გაგრძელება · halt-ის მოხსნა. ჩანს სტატუსი, ღია პოზიცია, დღის P&L, ლოგი, გარიგებები და სიგნალები. ყოველ 3 წამში ახლდება. |
| 🧠 სტრატეგიები | სტრატეგიების ფაილები, ახლის შექმნა შაბლონიდან, კოდის ნახვა. კოდს VS Code-ში წერ. |
| 📖 ინსტრუქცია | ეს ფაილი |

- Live config-ის ჩართვა პანელიდანაც დადასტურებას ითხოვს (checkbox).
- პანელი მხოლოდ შენს Mac-ზე იხსნება (localhost): ქსელიდან მასთან ვერავინ შევა.
- პანელი იმავე ბრძანებებს იყენებს, რაც ქვემოთაა აღწერილი. შედეგიც იგივეა.

### ხელით დაყენება (Terminal)

```bash
git clone https://github.com/luka-mdinaradze/test.git
cd test
git checkout claude/optimistic-curie-o1ogja

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-bot.txt

python3 -m pytest tests/test_tradebot.py tests/test_dashboard.py -q   # 33 ტესტი. ყველა უნდა იყოს passed
```

## 2. ოთხი ნაბიჯი, ამ რიგით

| ნაბიჯი | რა ხდება | ბრძანება |
|---|---|---|
| **1. Backtest** | ისტორიულ მონაცემზე, ინტერნეტის გარეშე | `python3 -m tradebot backtest --config configs/backtest_btc.toml` |
| **2. Paper** | რეალური ფასები, სიმულირებული შესრულება, ანგარიში არ სჭირდება | `python3 -m tradebot run --config configs/paper_btc_binance.toml` |
| **3. Testnet / IB paper** | რეალური ორდერები ტესტ-ანგარიშზე | `python3 -m tradebot run --config configs/live_btc_binance_testnet.toml --i-understand-live` |
| **4. Live** | რეალური ფული. მხოლოდ 1–3 ნაბიჯის შემდეგ | იგივე, `sandbox = false`-ით (ან IB live port-ით) |

- **1. Backtest.** იყენებს `examples/data`-ში ჩადებულ BTC-ის 15-წუთიან მონაცემს (2025-ის პირველი ნახევარი). შედეგი იწერება `runs/.../backtest/`-ში: `trades.csv`, `signals.csv`, `summary.json`, `equity.csv`. **Backtest არ არის რეალური შედეგი.**
- **2. Paper.** იღებს Binance-ის საჯარო ფასებს, API key არ სჭირდება. თუ Binance შენს ქვეყანაში არ მუშაობს, config-ში შეცვალე `exchange = "bybit"` ან `"okx"`.
- **3. Testnet.** უსაფრთხო ტესტია, ფული არ იხარჯება.
- **4. Live.** იგივე config, ოღონდ `sandbox = false` (ან IB-ის live port).

Live რეჟიმი **არ გაეშვება** `--i-understand-live` ფლაგის გარეშე. გაშვებამდე 5 წამს იცდის, რომ Ctrl+C-ით შეაჩერო.

### MNQ / NQ (Interactive Brokers)

1. დააყენე **TWS** ან **IB Gateway** (Mac-ზე მუშაობს).
2. TWS-ში: *Configure → API → Settings*. მონიშნე "Enable ActiveX and Socket Clients" და შეამოწმე port: **7497** paper-ისთვის (Gateway paper: 4002).
3. CME მონაცემის გამოწერა (market data subscription) IB-ის ანგარიშზე.
4. Paper: `python3 -m tradebot run --config configs/paper_mnq_ibkr.toml`. ფასები IB-დანაა, შესრულება სიმულირებული.
5. IB paper ანგარიშზე რეალური ორდერები: `configs/live_mnq_ibkr_paper_account.toml`.

**შეხსენება კვლევიდან:** $5,000-ზე 0.5% რისკით ($25) ერთი MNQ კონტრაქტი მხოლოდ დაახლოებით 11 პუნქტამდე სტოპს ეტევა. უფრო ფართო სტოპზე პლატფორმა გარიგებაზე **უარს იტყვის** და ამას `signals.csv`-ში ჩაწერს. ზომას არასოდეს დაამრგვალებს ზემოთ.

### API key-ები (კრიპტო)

Key **არასოდეს** იწერება config ფაილში, მხოლოდ terminal-ში:

```bash
export TRADEBOT_API_KEY="..."
export TRADEBOT_API_SECRET="..."
python3 -m tradebot check --config configs/live_btc_binance_testnet.toml   # აჩვენებს ბალანსს და პოზიციას
```

## 3. სტრატეგიის დაწერა (აქ ვეტყვით ბოტს, რა ქნას)

```bash
python3 -m tradebot new-strategy my_idea        # ქმნის tradebot/strategies/my_idea.py-ს შაბლონიდან
```

config-ში ჩაწერე `[strategy] name = "my_idea"`. ფაილში მხოლოდ ეს ნაწილი იცვლება:

```python
def on_bar(self, ctx):
    # ctx.close / ctx.high / ctx.low / ctx.volume — მხოლოდ დახურული ბარები, ბოლო = უახლესი
    # ctx.position — ღია პოზიცია ან None
    atr = ind.atr(ctx.high, ctx.low, ctx.close, 14).iloc[-1]
    if ctx.position is None and <შესვლის პირობა>:
        return Signal.long(stop=1.5 * atr, target=3.0 * atr, max_bars=40, reason="რატომ")
    if ctx.position is not None and <გამოსვლის პირობა>:
        return Exit("რატომ")
    return None
```

- **`stop` და `target` მანძილებია** ფასის ერთეულებში, შესრულების ფასიდან. Stop სავალდებულოა: პოზიცია stop-ის გარეშე არ იხსნება.
- **პარამეტრები:** ყველა რიცხვი იწერება `defaults`-ში და იცვლება config-ის `[strategy.params]`-ში, კოდის შეცვლის გარეშე.
- **მზა ინდიკატორები** (`tradebot/indicators.py`): `sma`, `ema`, `atr`, `rsi`, `zscore`, `highest`, `lowest`, `crossed_above`, `crossed_below`.
- **`ctx.note("...")`** ტექსტს ამატებს სიგნალის ბეჭდვის "MARKET REGIME" ველში.
- **ალბათობა:** `probability=` მიუთითე მხოლოდ მაშინ, თუ შენი მოდელი out-of-sample-ზე კალიბრირებული და შემოწმებულია. სხვა შემთხვევაში პლატფორმა წერს "unavailable" და ციფრს არ იგონებს.
- **მაგალითი:** `tradebot/strategies/example_ema_cross.py` მხოლოდ API-ის საჩვენებლადაა. ის **არ არის შემოწმებული** და backtest-ში წააგებს.

## 4. რისკის პარამეტრები (`[account]`)

| პარამეტრი | ნაგულისხმევი | რას აკეთებს |
|---|---|---|
| `risk_pct` | 0.5 | % ანგარიშიდან ერთ გარიგებაზე (შესვლიდან სტოპამდე, ხარჯების ჩათვლით) |
| `max_risk_pct` | 1.0 | ზედა ზღვარი. 2%-ზე მეტს config არ მიიღებს |
| `daily_loss_limit_pct` | 2.0 | დღეში −2%: იმ დღეს ახალი გარიგება აღარ იხსნება |
| `max_consecutive_losses` | 4 | ზედიზედ 4 წაგება: პაუზა დღის ბოლომდე |
| `max_trades_per_day` | 10 | გარიგებების დღიური ლიმიტი |
| `max_drawdown_pct` | 15.0 | პიკიდან −15%: **სრული გაჩერება**. ადამიანმა უნდა გაასუფთაოს |

სესია (`[session]`):
- `trade_start` / `trade_end`: როდის შეიძლება შესვლა;
- `flatten_at`: ყოველდღე ყველაფერი იხურება ამ დრომდე;
- `weekdays`: რომელ დღეებში ვაჭრობს;
- `timezone`: ბირჟის დრო (MNQ-ისთვის `America/New_York`).

## 5. უსაფრთხოება

- **Stop ყოველთვის ბირჟაზე დგას**, გახსნის მომენტიდან. თუ ბირჟა stop-ს არ მიიღებს, ბოტი პოზიციას მაშინვე ხურავს. თუ ბოტი ან Mac გაითიშა, stop მაინც რჩება.
- **ბირჟა არის ჭეშმარიტების წყარო.** გადატვირთვისას ბოტი ბირჟას ეკითხება, რა არის რეალურად ღია. თუ ბირჟაზე ისეთი პოზიციაა, რომელიც ბოტს არ გაუხსნია, ბოტი ჩერდება და გელოდება.
- **Kill switch:** პანელში ⏹, ან `python3 -m tradebot stop --config ...`. აჩერებს ახალ გარიგებებს და ღია პოზიციას ხურავს. `resume` (პანელში ↺) ხსნის.
- **სრული გაჩერების (halt) მოხსნა** მხოლოდ ხელით, მიზეზის ნახვის შემდეგ: `reset-halt --config ...`.
- **MacBook-ის ძილი:** თუ Mac დაიძინებს, ბოტი ჩერდება (stop-ები ბირჟაზე რჩება). გაშვება ძილის გარეშე: `caffeinate -i python3 -m tradebot run --config ...`. ხანგრძლივი მუშაობისთვის ჯობს VPS.
- **სტატუსი:** `python3 -m tradebot status --config ...`. ლოგი: `runs/<ბოტი>/bot.log`, გარიგებები: `trades.csv`.

## 6. რა არის შემოწმებული და რა არა

**შემოწმებულია (33 ტესტი):**
- paper შესრულების წესები: შემდეგი ბარის open, ერთ ბარში stop-ისა და target-ის შეხებისას stop, gap, target trade-through, ხარჯები;
- ზომა და უარი ზედმეტ ზომაზე;
- ყველა რისკის ლიმიტი, kill switch, სესიის წესები;
- CSV მონაცემი და ჯერ დაუხურავი ბარის გამორიცხვა;
- სრული backtest და live ციკლი;
- მდგომარეობის აღდგენა და reconcile;
- ccxt-ის და IBKR-ის ორდერების ლოგიკა ყალბ (fake) ბირჟაზე;
- პანელი: Backtest ღილაკი იმავე შედეგს იძლევა, რასაც ბრძანება; ღილაკები რეჟიმის მიხედვით ირთვება; ბოტის პროცესი ირთვება და ჩერდება; სტრატეგიის ფაილში შეცდომა პანელს არ აგდებს.

**შეუმოწმებელია:** რეალურ ბირჟასთან ან IB-თან კავშირი. ის გარემო, სადაც ეს დაიწერა, ბირჟებს ვერ წვდება. ამიტომ ccxt და IBKR ადაპტერები **ჯერ testnet-ზე ან IB paper ანგარიშზე** უნდა გაეშვას.

## 7. ბრძანებები

```bash
python3 -m tradebot backtest     --config FILE [--csv DATA.csv] [--verbose]
python3 -m tradebot run          --config FILE [--i-understand-live] [--max-bars N]
python3 -m tradebot check        --config FILE      # config + სტრატეგია (+ ბროკერთან კავშირი live-ზე)
python3 -m tradebot status       --config FILE
python3 -m tradebot stop|resume  --config FILE      # kill switch
python3 -m tradebot reset-halt   --config FILE
python3 -m tradebot new-strategy NAME
python3 -m tradebot dashboard    [--port N]         # მართვის პანელი ბრაუზერში
```

| ფაილი | რა არის |
|---|---|
| `tradebot/strategies/template.py` | **სტრატეგიის შაბლონი: აქ ჩაიწერება წესები** |
| `tradebot/runner.py` | მთავარი ციკლი (ერთი backtest-ის, paper-ის და live-ისთვის) |
| `tradebot/risk.py` | ზომა და რისკის ლიმიტები |
| `tradebot/session.py` | სესიის წესები |
| `tradebot/brokers/` | `paper.py`, `ccxt_broker.py`, `ibkr.py` |
| `tradebot/feeds.py` | მონაცემის წყაროები: CSV, ccxt, IBKR |
| `tradebot/indicators.py` | ინდიკატორები |
| `configs/*.toml` | მზა კონფიგურაციები |
| `start_dashboard.command` | პანელის გამშვები Mac-ზე (ორჯერ დააწკაპუნე) |
| `tradebot/dashboard.py` | მართვის პანელი |
| `tradebot/control.py` | ბოტის პროცესის ჩართვა და გაჩერება |

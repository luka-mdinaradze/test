"""tradebot control panel — one window for backtests, the bot, strategies and the guide.

    python3 -m tradebot dashboard          (opens http://localhost:8501)

Runs locally. The bot itself runs as a separate background process, so closing
this window does not stop it; use the Stop buttons for that.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tradebot import config as cfgmod  # noqa: E402
from tradebot import control  # noqa: E402
from tradebot.feeds import load_csv  # noqa: E402
from tradebot.runner import Runner  # noqa: E402
from tradebot.state import StateStore  # noqa: E402
from tradebot.strategy import create_strategy_file, list_strategies, load_strategy  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(os.environ.get("TRADEBOT_CONFIG_DIR", PROJECT / "configs"))
DATA_DIR = PROJECT / "examples" / "data"
STRAT_DIR = Path(__file__).resolve().parent / "strategies"
MODE_LABEL = {"backtest": "Backtest", "paper": "Paper (სიმულაცია)", "live": "LIVE (რეალური ორდერები)"}

st.set_page_config(page_title="tradebot", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------- chart styling

def theme() -> dict:
    """Validated categorical steps per mode (long = slot 1 blue, short = slot 2 orange)."""
    dark = getattr(st.context.theme, "type", None) == "dark"
    return {"s1": "#3987e5" if dark else "#2a78d6", "s2": "#d95926" if dark else "#eb6834",
            "muted": "#898781", "grid": "#2c2c2a" if dark else "#e1e0d9", "axis": "#383835" if dark else "#c3c2b7",
            "ink2": "#c3c2b7" if dark else "#52514e", "surface": "#1a1a19" if dark else "#fcfcfb"}


def styled(chart: alt.Chart) -> alt.Chart:
    t = theme()
    return (chart.configure(background="transparent")
            .configure_view(strokeWidth=0)
            .configure_axisX(grid=False)
            .configure_axis(gridColor=t["grid"], domainColor=t["axis"], tickColor=t["axis"],
                            labelColor=t["muted"], titleColor=t["ink2"], labelFontSize=11, titleFontSize=11,
                            titleFontWeight="normal")
            .configure_legend(labelColor=t["ink2"], titleColor=t["ink2"], orient="top", title=None))


def line_with_crosshair(df: pd.DataFrame, x: str, y: str, color: str, y_title: str, fmt: str,
                        baseline: float | None = None, height: int = 260) -> alt.Chart:
    """2px line; a vertical hairline snaps to the nearest x and carries the tooltip."""
    t = theme()
    base = alt.Chart(df).encode(x=alt.X(f"{x}:T", title=None))
    line = base.mark_line(strokeWidth=2, color=color).encode(
        y=alt.Y(f"{y}:Q", title=y_title, scale=alt.Scale(zero=False)))
    near = alt.selection_point(nearest=True, on="pointerover", fields=[x], empty=False, clear="pointerout")
    selectors = base.mark_point().encode(y=f"{y}:Q", opacity=alt.value(0)).add_params(near)
    rule = base.mark_rule(color=t["muted"], strokeWidth=1).encode(
        opacity=alt.condition(near, alt.value(0.8), alt.value(0)),
        tooltip=[alt.Tooltip(f"{y}:Q", title=y_title, format=fmt),
                 alt.Tooltip(f"{x}:T", title="დრო", format="%Y-%m-%d %H:%M")])
    dot = base.mark_point(size=70, filled=True, color=color, stroke=t["surface"], strokeWidth=2).encode(
        y=f"{y}:Q", opacity=alt.condition(near, alt.value(1), alt.value(0)))
    layers = [line, selectors, rule, dot]
    if baseline is not None:
        layers.insert(0, alt.Chart(pd.DataFrame({"b": [baseline]})).mark_rule(
            color=t["axis"], strokeDash=[4, 4]).encode(y="b:Q"))
    return styled(alt.layer(*layers).properties(height=height))


def price_with_trades(price: pd.DataFrame, trades: pd.DataFrame, height: int = 320) -> alt.Chart:
    t = theme()
    p = price.reset_index().rename(columns={"ts": "time"})
    line = alt.Chart(p).mark_line(strokeWidth=1.5, color=t["muted"]).encode(
        x=alt.X("time:T", title=None), y=alt.Y("close:Q", title="ფასი", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("close:Q", title="close", format=",.2f"),
                 alt.Tooltip("time:T", title="დრო", format="%Y-%m-%d %H:%M")])
    if trades.empty:
        return styled(line.properties(height=height))
    tr = trades.copy()
    tr["time"] = pd.to_datetime(tr["opened_at"], utc=True)
    tr["მიმართულება"] = tr["side"].map({"LONG": "Long ▲", "SHORT": "Short ▼"})
    dom, rng = ["Long ▲", "Short ▼"], [t["s1"], t["s2"]]
    pts = alt.Chart(tr).mark_point(filled=True, size=90, stroke=t["surface"], strokeWidth=1.5, opacity=1).encode(
        x="time:T", y=alt.Y("entry:Q"),
        color=alt.Color("მიმართულება:N", scale=alt.Scale(domain=dom, range=rng)),
        shape=alt.Shape("მიმართულება:N", scale=alt.Scale(domain=dom, range=["triangle-up", "triangle-down"])),
        tooltip=[alt.Tooltip("pnl:Q", title="P&L $", format="+,.2f"), alt.Tooltip("r:Q", title="R", format="+.2f"),
                 alt.Tooltip("მიმართულება:N"), alt.Tooltip("entry:Q", title="შესვლა", format=",.2f"),
                 alt.Tooltip("exit:Q", title="გასვლა", format=",.2f"), alt.Tooltip("exit_reason:N", title="გასვლის მიზეზი"),
                 alt.Tooltip("time:T", title="გახსნა", format="%Y-%m-%d %H:%M")])
    return styled(alt.layer(line, pts).properties(height=height))


# --------------------------------------------------------------------------- helpers

@st.cache_data(show_spinner=False)
def cached_csv(path: str, mtime: float, tf: int) -> pd.DataFrame:
    return load_csv(path, tf)


def config_files() -> list[Path]:
    return sorted(CONFIG_DIR.glob("*.toml"))


def downsample(df: pd.DataFrame, tf: int, max_points: int = 2000) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = math.ceil(len(df) / max_points) * tf
    return df.resample(f"{step}min").agg({"close": "last"}).dropna()


def param_inputs(defaults: dict, current: dict, key: str) -> dict:
    out = {}
    cols = st.columns(4)
    for i, (k, v) in enumerate(defaults.items()):
        val = current.get(k, v)
        with cols[i % 4]:
            if isinstance(v, bool):
                out[k] = st.checkbox(k, value=bool(val), key=f"{key}-{k}")
            elif isinstance(v, int):
                out[k] = int(st.number_input(k, value=int(val), step=1, key=f"{key}-{k}"))
            elif isinstance(v, float):
                out[k] = float(st.number_input(k, value=float(val), step=0.1, format="%.4g", key=f"{key}-{k}"))
            else:
                out[k] = st.text_input(k, value=str(val), key=f"{key}-{k}")
    return out


def esc(text: str) -> str:
    """Streamlit renders $...$ as LaTeX; keep dollar amounts literal."""
    return str(text).replace("$", "\\$")


def esc_markdown(text: str) -> str:
    """esc() outside code spans and fenced blocks, which show `$` as typed."""
    parts = re.split(r"(```.*?```|`[^`\n]*`)", text, flags=re.S)
    return "".join(p if p.startswith("`") else esc(p) for p in parts)


def money(x) -> str:
    if x is None:
        return "—"
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


# --------------------------------------------------------------------------- sidebar

st.sidebar.title("📈 tradebot")
files = config_files()
if not files:
    st.error(f"{CONFIG_DIR}-ში config ფაილი ვერ ვიპოვე.")
    st.stop()
choice = st.sidebar.selectbox("კონფიგურაცია", files, format_func=lambda p: p.stem, key="config")
try:
    cfg = cfgmod.load(choice)
except ValueError as e:
    st.sidebar.error(str(e))
    st.stop()
st.sidebar.markdown(
    f"**{cfg.bot.name}**  \n"
    f"რეჟიმი: **{MODE_LABEL[cfg.bot.mode]}**  \n"
    f"ბროკერი / მონაცემი: `{cfg.market.broker}` / `{cfg.market.feed}`  \n"
    f"ინსტრუმენტი: `{cfg.market.symbol}` · {cfg.market.timeframe}  \n"
    f"სტრატეგია: `{cfg.strategy.name}`  \n"
    f"რისკი: {cfg.account.risk_pct:g}% გარიგებაზე (max {cfg.account.max_risk_pct:g}%)")


def bot_state(cfg) -> tuple[int | None, dict, dict]:
    pid = control.running_pid(cfg)
    state = StateStore(cfg.state_dir).load() if cfg.state_dir.exists() else {}
    return pid, state, state.get("governor") or {}


def status_badge(pid: int | None) -> str:
    return f":green-badge[● მუშაობს · PID {pid}]" if pid else ":gray-badge[○ გაჩერებულია]"


if cfg.bot.mode != "backtest":
    @st.fragment(run_every="3s")
    def sidebar_status():
        pid, _, gov = bot_state(cfg)
        if gov.get("halted_reason"):
            st.error("⛔ HALT: " + esc(gov["halted_reason"]))
        st.markdown("ბოტი: " + status_badge(pid))

    with st.sidebar:
        sidebar_status()
st.sidebar.caption("Backtest და paper შედეგი არ არის რეალური შედეგი.")

tab_bt, tab_bot, tab_strat, tab_help = st.tabs(["📊 Backtest", "🤖 ბოტი", "🧠 სტრატეგიები", "📖 ინსტრუქცია"])


# --------------------------------------------------------------------------- backtest

with tab_bt:
    st.subheader("Backtest")
    strategies = list_strategies(STRAT_DIR)
    c1, c2, c3 = st.columns([2, 3, 1])
    with c1:
        strat_name = st.selectbox("სტრატეგია", strategies,
                                  index=strategies.index(cfg.strategy.name) if cfg.strategy.name in strategies else 0,
                                  key="bt-strategy")
    datasets = sorted(DATA_DIR.glob("*.csv"))
    if cfg.market.csv_path and Path(cfg.market.csv_path).exists() and Path(cfg.market.csv_path) not in datasets:
        datasets.insert(0, Path(cfg.market.csv_path))
    uploaded = st.session_state.get("uploaded_csv")
    if uploaded:
        datasets.insert(0, Path(uploaded))
    with c2:
        data_path = st.selectbox("მონაცემი (CSV: timestamp, open, high, low, close, volume)", datasets,
                                 format_func=lambda p: p.name, key="bt-data") if datasets else None
    with c3:
        risk = st.number_input("რისკი %", 0.1, cfg.account.max_risk_pct, float(cfg.account.risk_pct), 0.1, key="bt-risk")
    up = st.file_uploader("ან ატვირთე შენი CSV", type="csv", key="bt-upload")
    if up is not None:
        dst = PROJECT / "runs" / "uploads" / up.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(up.getvalue())
        if st.session_state.get("uploaded_csv") != str(dst):
            st.session_state["uploaded_csv"] = str(dst)
            st.rerun()

    if data_path is None:
        st.info("მონაცემი არ არის: ჩადე CSV `examples/data/`-ში ან ატვირთე.")
    else:
        try:
            df_all = cached_csv(str(data_path), data_path.stat().st_mtime, cfg.tf_minutes)
        except ValueError as e:
            st.error(str(e))
            df_all = None
        if df_all is not None:
            lo, hi = df_all.index[0].date(), df_all.index[-1].date()
            d1, d2, _ = st.columns([1, 1, 2])
            start = d1.date_input("დან", lo, min_value=lo, max_value=hi, key="bt-start")
            end = d2.date_input("მდე", hi, min_value=lo, max_value=hi, key="bt-end")
            try:
                strat_cls = type(load_strategy(strat_name))
            except Exception as e:      # a typo in a strategy file must not take the panel down
                st.error(f"სტრატეგიის ფაილში შეცდომაა ({strat_name}.py): {e}")
                st.stop()
            current = cfg.strategy.params if strat_name == cfg.strategy.name else {}
            with st.expander(f"სტრატეგიის პარამეტრები ({strat_name})", expanded=False):
                params = param_inputs(strat_cls.defaults, current, f"p-{strat_name}") if strat_cls.defaults else {}
                if not strat_cls.defaults:
                    st.caption("ამ სტრატეგიას პარამეტრები არ აქვს.")

            if st.button("▶ Backtest-ის გაშვება", type="primary", key="run_backtest"):
                c = copy.deepcopy(cfg)
                c.bot.mode, c.market.broker, c.market.feed = "backtest", "paper", "csv"
                c.market.csv_path, c.market.start, c.market.end = str(data_path), str(start), str(end)
                c.account.risk_pct = float(risk)
                c.strategy.name, c.strategy.params = strat_name, params
                with st.spinner("ვითვლი… (ნახევარი წლის 15-წუთიანი ბარები ~15 წამი)"):
                    try:
                        r = Runner(c, state_dir=c.state_dir / "dashboard", printer=lambda *_: None)
                        summary = r.backtest()
                        st.session_state["bt"] = {
                            "summary": summary, "trades": pd.DataFrame(r.trades),
                            "equity": pd.DataFrame(r.equity_curve, columns=["ts", "equity", "in_position"]),
                            "start_equity": c.account.starting_equity, "label": f"{strat_name} · {data_path.name}",
                            "data": str(data_path), "range": (str(start), str(end)), "tf": c.tf_minutes}
                    except Exception as e:      # show any strategy error in the panel instead of crashing it
                        st.session_state.pop("bt", None)
                        st.exception(e)

    res = st.session_state.get("bt")
    if res:
        s, trades = res["summary"], res["trades"]
        st.markdown(f"#### შედეგი: {res['label']}")
        if s["trades"] == 0:
            st.info("სტრატეგიას არცერთი გარიგება არ გაუკეთებია. ცარიელი შაბლონი ასეც უნდა იქცეოდეს, სანამ წესებს არ ჩაწერ.")
        row1, row2 = st.columns(3), st.columns(3)
        ci = s.get("expectancy_r_95ci")
        row1[0].metric("გარიგებები", s["trades"], border=True)
        row1[1].metric("მოლოდინი (R / გარიგება)", "—" if s["expectancy_r"] is None else f"{s['expectancy_r']:+.3f}",
                       help=f"95% ინტერვალი: {ci[0]:+.3f} … {ci[1]:+.3f} R" if ci else None, border=True)
        row1[2].metric("მოგებიანი გარიგებები", "—" if s["win_rate_pct"] is None else f"{s['win_rate_pct']:.1f}%",
                       border=True)
        row2[0].metric("წმინდა P&L", money(s["net_pnl"]), f"{s['return_pct']:+.2f}%", border=True)
        row2[1].metric("Profit factor", "—" if s["profit_factor"] is None else f"{s['profit_factor']:.2f}", border=True)
        row2[2].metric("Max drawdown", f"{s['max_drawdown_pct']:.1f}%", border=True)
        if ci:
            verdict = ("ინტერვალი მთლიანად ნულს ზემოთაა" if ci[0] > 0 else
                       "ინტერვალი მთლიანად ნულს ქვემოთაა: წამგებიანია" if ci[1] < 0 else
                       "ინტერვალი ნულს მოიცავს: უპირატესობა დადასტურებული არ არის")
            st.caption(f"მოლოდინის 95% ინტერვალი: {ci[0]:+.3f} … {ci[1]:+.3f} R. {verdict}. "
                       "ერთი პერიოდის backtest არ არის მტკიცებულება.")
        if s.get("halted"):
            st.warning(f"⛔ რისკის მართვამ ვაჭრობა შეაჩერა {s['halted']['at']}: {esc(s['halted']['reason'])}. "
                       "ამის შემდეგ ყველა სიგნალი დაიბლოკა (როგორც live-ში).")
        eq = res["equity"].copy()
        eq["ts"] = pd.to_datetime(eq["ts"], utc=True)
        eq = eq.set_index("ts")
        st.markdown("**ანგარიშის ბალანსი**")
        eq_ds = eq[["equity"]].resample(f"{max(1, math.ceil(len(eq) / 2000)) * res['tf']}min").last().dropna().reset_index()
        st.altair_chart(line_with_crosshair(eq_ds, "ts", "equity", theme()["s1"], "ბალანსი $", "$,.2f",
                                            baseline=res["start_equity"]), width="stretch", theme=None)
        st.markdown("**ფასი და გარიგებები** (▲ long · ▼ short; მიიტანე მაუსი გარიგებაზე)")
        px = cached_csv(res["data"], Path(res["data"]).stat().st_mtime, res["tf"])
        px = px[(px.index >= pd.Timestamp(res["range"][0], tz="UTC")) &
                (px.index < pd.Timestamp(res["range"][1], tz="UTC") + pd.Timedelta(days=1))]
        st.altair_chart(price_with_trades(downsample(px, res["tf"]), trades), width="stretch", theme=None)
        a, b = st.columns([3, 2])
        with a:
            st.markdown("**გარიგებები**")
            if not trades.empty:
                st.dataframe(trades[["opened_at", "closed_at", "side", "qty", "entry", "exit", "pnl", "r", "bars",
                                     "exit_reason", "entry_reason"]], hide_index=True, height=300)
                st.download_button("⬇ trades.csv", trades.to_csv(index=False), "trades.csv", "text/csv")
        with b:
            st.markdown("**სიგნალები**")
            st.dataframe(pd.DataFrame({
                "": ["სულ", "გაგზავნილი", "დაბლოკილი (რისკი / სესია)", "უარი (ზომა)", "ბროკერის შეცდომა"],
                "რაოდენობა": [s["signals_total"], s["signals_sent"], s["signals_blocked"],
                              s["signals_rejected_size"], s["signals_broker_errors"]]}), hide_index=True)
            if s.get("exit_reasons"):
                st.markdown("**გასვლის მიზეზები**")
                st.dataframe(pd.DataFrame(list(s["exit_reasons"].items()), columns=["მიზეზი", "რაოდენობა"]),
                             hide_index=True)
        st.caption("Backtest არ არის რეალური შედეგი. შესრულება: შემდეგი ბარის open, ერთ ბარში stop და target "
                   "ორივეს შეხებისას stop, gap-ზე open-ის ფასით, ხარჯები ორივე მხარეს.")


# --------------------------------------------------------------------------- bot

with tab_bot:
    st.subheader("ბოტი")
    if cfg.bot.mode == "backtest":
        st.info("ეს backtest-ის config-ია. ბოტის გასაშვებად მარცხნივ აირჩიე **paper** ან **live** config "
                "(მაგ. `paper_btc_binance`).")
    else:
        if cfg.bot.mode == "live":
            st.error("**LIVE რეჟიმი**: ბოტი ბროკერთან რეალურ ორდერებს აგზავნის. "
                     + ("ეს testnet ან paper ანგარიშია." if (cfg.market.broker == "ccxt" and cfg.ccxt.sandbox) or
                        (cfg.market.broker == "ibkr" and cfg.ibkr.port in (7497, 4002)) else "**ეს რეალური ფულია.**"))

        @st.fragment(run_every="3s")
        def bot_panel():
            pid, state, gov = bot_state(cfg)
            pos = state.get("position")
            killed = (cfg.state_dir / "STOP").exists()
            st.markdown("##### " + status_badge(pid))
            last = state.get("last_bar") or ""
            k = st.columns(3)
            k[0].metric("ბოლო დახურული ბარი", last[5:16].replace("T", " ") or "—",
                        help=f"{last} (UTC)" if last else None, border=True)
            k[1].metric("დღის P&L", money(gov.get("day_pnl")) if gov else "—", border=True)
            k[2].metric("დღეს გარიგება", gov.get("trades_today", "—") if gov else "—", border=True)
            if gov.get("halted_reason"):
                st.error("⛔ HALT: " + esc(gov["halted_reason"]))
            elif gov.get("paused_reason"):
                st.warning("⏸ დღის პაუზა: " + esc(gov["paused_reason"]))
            if killed:
                st.warning("🛑 Kill switch ჩართულია: ახალი გარიგება არ გაიხსნება, ღია პოზიცია დაიხურება და ბოტი გაჩერდება.")
            if pos:
                side = "LONG ▲" if pos["side"] > 0 else "SHORT ▼"
                text = (f"ღია პოზიცია: **{side} {pos['qty']:g}** · შესვლა {pos['entry_price']:,.2f} · "
                        f"stop {pos['stop_price']:,.2f}")
                if pos.get("target_price"):
                    text += f" · target {pos['target_price']:,.2f}"
                st.success(text + f" · {pos.get('bars_held', 0)} ბარი")
            elif pid:
                st.caption("ღია პოზიცია არ არის. ბოტი შემდეგი დახურული ბარის სიგნალს ელოდება.")

            b = st.columns(5)
            confirm = True
            if cfg.bot.mode == "live" and not pid:
                confirm = st.checkbox("ვიცი, რომ ეს რეალურ ორდერებს აგზავნის და ჯერ testnet / paper ანგარიშზე შევამოწმე",
                                      key="live-confirm")
            if b[0].button("▶ ჩართვა", type="primary", disabled=bool(pid) or not confirm, key="bot-start",
                           width="stretch"):
                try:
                    control.start(cfg, live_confirmed=confirm)
                    st.toast("ბოტი ჩაირთო")
                except RuntimeError as e:
                    st.error(str(e))
                st.rerun(scope="fragment")
            if b[1].button("⏹ გაჩერება", disabled=not pid, key="bot-stop", width="stretch",
                           help="Kill switch: ახალი გარიგება აღარ გაიხსნება, ღია პოზიცია დაიხურება შემდეგ ბარზე, მერე ბოტი ჩერდება"):
                control.stop_graceful(cfg)
                st.rerun(scope="fragment")
            if b[2].button("⛔ ახლავე", disabled=not pid, key="bot-kill", width="stretch",
                           help="Ctrl+C ახლავე. სტეიტი ინახება, ბირჟაზე უკვე მდგარი stop რჩება"):
                control.stop_force(cfg)
                st.rerun(scope="fragment")
            if b[3].button("↺ გაგრძელება", disabled=not killed, key="bot-resume", width="stretch",
                           help="Kill switch-ის მოხსნა: ბოტს ისევ შეეძლება გარიგების გახსნა"):
                (cfg.state_dir / "STOP").unlink(missing_ok=True)
                st.rerun(scope="fragment")
            if b[4].button("Halt-ის მოხსნა", disabled=not gov.get("halted_reason"), key="bot-reset", width="stretch",
                           help="მხოლოდ მიზეზის ნახვის შემდეგ"):
                d = StateStore(cfg.state_dir).load()
                d["governor"]["halted_reason"], d["governor"]["peak_equity"] = "", 0.0
                (cfg.state_dir / "state.json").write_text(json.dumps(d, indent=2, default=str))
                st.rerun(scope="fragment")

            log = control.tail(cfg.state_dir / "bot.log", 40) or control.tail(cfg.state_dir / "console.log", 40)
            with st.expander("ლოგი (ბოლო 40 ხაზი)", expanded=bool(pid)):
                st.code(log or "ჯერ ცარიელია", language="text")
            tpath = cfg.state_dir / "trades.csv"
            if tpath.exists():
                tr = pd.read_csv(tpath)
                if len(tr) and "pnl" in tr:
                    tr["closed_at"] = pd.to_datetime(tr["closed_at"], utc=True, errors="coerce")
                    cum = tr.dropna(subset=["closed_at"]).assign(cum=lambda d: d["pnl"].cumsum())[["closed_at", "cum"]]
                    if len(cum) > 1:
                        st.markdown("**ჯამური P&L**")
                        st.altair_chart(line_with_crosshair(cum, "closed_at", "cum", theme()["s1"], "P&L $", "+$,.2f",
                                                            baseline=0.0, height=220), width="stretch", theme=None)
                    st.markdown("**გარიგებები**")
                    st.dataframe(tr.iloc[::-1], hide_index=True, height=240)
            spath = cfg.state_dir / "signals.csv"
            if spath.exists():
                st.markdown("**ბოლო სიგნალები** (მათ შორის დაბლოკილი)")
                st.dataframe(pd.read_csv(spath).tail(30).iloc[::-1], hide_index=True, height=240)

        bot_panel()
        st.caption("ბოტი ცალკე პროცესად მუშაობს: ამ ფანჯრის დახურვა მას არ აჩერებს. Mac-ის ძილისას ბოტი "
                   "ჩერდება (ბირჟაზე მდგარი stop რჩება). ძილის გარეშე: `caffeinate -i`.")


# --------------------------------------------------------------------------- strategies

with tab_strat:
    st.subheader("სტრატეგიები")
    st.markdown("სტრატეგია ერთი ფაილია `tradebot/strategies/`-ში. აქ ჩაიწერება, **რა ქნას ბოტმა**. "
                "კოდი VS Code-ში შეცვალე, მერე აქ Backtest-ში გაუშვი.")
    names = list_strategies(STRAT_DIR)
    a, b = st.columns([1, 2])
    with a:
        sel = st.radio("ფაილები", names, key="strat-sel")
        new = st.text_input("ახალი სტრატეგიის სახელი (ლათინურად, მაგ. my_idea)", key="strat-new")
        if st.button("＋ შექმნა შაბლონიდან", key="strat-create", disabled=not new):
            try:
                p = create_strategy_file(new.strip(), STRAT_DIR)
                st.success(f"შეიქმნა: tradebot/strategies/{p.name}. გახსენი VS Code-ში და ჩაწერე წესები.")
                st.rerun()
            except ValueError as e:
                st.error(str(e))
    with b:
        if sel:
            try:
                s_obj = load_strategy(sel)
                st.caption(f"კლასი `{type(s_obj).__name__}` · warmup {s_obj.warmup} ბარი · "
                           f"პარამეტრები: {json.dumps(s_obj.params, ensure_ascii=False)}")
            except Exception as e:
                st.error(f"ფაილში შეცდომაა: {e}")
            st.code((STRAT_DIR / f"{sel}.py").read_text(), language="python", line_numbers=True)


# --------------------------------------------------------------------------- guide

with tab_help:
    guide = PROJECT / "TRADEBOT.md"
    st.markdown(esc_markdown(guide.read_text()) if guide.exists() else "TRADEBOT.md ვერ ვიპოვე.")

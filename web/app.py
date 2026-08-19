"""Atlas web app — FastAPI backend with user login and the NSE screener.

Run:  uvicorn web.app:app --reload --port 8000
      (or: python -m web.app)
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web import auth
from web.data_service import chain, expiries, get_screen, refresh

BASE = Path(__file__).parent
app = FastAPI(title="Atlas")

# Refusing to hand out the OTP is only half the fix; the operator also has to be
# told why login stopped working, or they will assume the app is broken.
OPEN_LOGIN_ERROR = (
    "This server is reachable from the network, so Atlas won't show login codes on "
    "screen — that would let anyone sign in as anyone. Set SMTP_HOST / SMTP_USER / "
    "SMTP_PASS to email codes, or ATLAS_ALLOWED_EMAILS to limit who can sign in."
)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")


# ---------------------------------------------------------------- auth dep
def current_user(request: Request) -> str | None:
    return auth.read_session(request.cookies.get(auth.COOKIE_NAME))


def require_user(request: Request) -> str:
    email = current_user(request)
    if not email:
        # Signal the route to redirect
        raise _NotAuthenticated()
    return email


class _NotAuthenticated(Exception):
    pass


@app.exception_handler(_NotAuthenticated)
async def _redirect_login(request: Request, exc: _NotAuthenticated):
    return RedirectResponse("/login", status_code=303)


# ---------------------------------------------------------------- pages
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if current_user(request):
        return RedirectResponse("/dashboard", status_code=303)
    return RedirectResponse("/login", status_code=303)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if current_user(request):
        return RedirectResponse("/dashboard", status_code=303)
    return templates.TemplateResponse(request, "login.html", {"error": None})


@app.post("/login")
async def login(request: Request, email: str = Form(...)):
    """Step 1: request an OTP for this email."""
    email = email.strip().lower()
    if not auth.valid_email(email):
        return templates.TemplateResponse(request, "login.html",
                                          {"error": "Enter a valid email address."})
    if not auth.email_allowed(email):
        # Same message either way — don't tell a stranger which emails exist.
        return templates.TemplateResponse(request, "login.html",
                                          {"error": "That email isn't allowed to sign in."})
    if auth.bound_publicly() and auth.login_is_open():
        return templates.TemplateResponse(request, "login.html", {"error": OPEN_LOGIN_ERROR})
    code = auth.generate_otp(email)
    sent, info = auth.send_otp_email(email, code)   # real email if SMTP set, else dev
    if auth.smtp_configured() and not sent:
        return templates.TemplateResponse(request, "login.html",
                                          {"error": f"Couldn't send the email ({info}). "
                                                    "Check your SMTP settings in .env."})
    return RedirectResponse(f"/verify?email={email}", status_code=303)


@app.get("/verify", response_class=HTMLResponse)
async def verify_page(request: Request, email: str = ""):
    if current_user(request):
        return RedirectResponse("/dashboard", status_code=303)
    if not auth.valid_email(email):
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(request, "verify.html", {
        "email": email, "error": None,
        "dev_code": auth.dev_code(email),           # withheld unless loopback-only
        "emailed": auth.smtp_configured(),
    })


@app.post("/verify")
async def verify(request: Request, email: str = Form(...), code: str = Form(...)):
    """Step 2: check the OTP and open a session."""
    email = email.strip().lower()
    if auth.verify_otp(email, code):
        resp = RedirectResponse("/dashboard", status_code=303)
        resp.set_cookie(auth.COOKIE_NAME, auth.make_session(email),
                        max_age=auth.SESSION_MAX_AGE, httponly=True, samesite="lax")
        return resp
    return templates.TemplateResponse(request, "verify.html", {
        "email": email, "error": "Invalid or expired code. Try again.",
        "dev_code": auth.dev_code(email), "emailed": auth.smtp_configured(),
    })


@app.get("/signup")
async def signup_redirect():
    return RedirectResponse("/login", status_code=303)


@app.get("/logout")
async def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(auth.COOKIE_NAME)
    return resp


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, email: str = Depends(require_user)):
    user = auth.get_user(email) or {"name": email}
    resp = templates.TemplateResponse(request, "dashboard.html", {"user": user})
    # never cache the app shell, so code updates always reach the browser
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


# ---------------------------------------------------------------- API
@app.get("/api/screen")
async def api_screen(request: Request, interval: int = 5, force: bool = False,
                     email: str = Depends(require_user)):
    df = refresh(interval) if force else get_screen(interval)
    if df.empty:
        return JSONResponse({"rows": [], "updated": None})
    df = df.reindex(df["conviction"].abs().sort_values(ascending=False).index)
    import datetime as _dt
    return JSONResponse({
        "rows": df.to_dict("records"),
        "count": len(df),
        "updated": _dt.datetime.now().strftime("%H:%M:%S"),
    })


@app.get("/api/expiries")
async def api_expiries(symbol: str, email: str = Depends(require_user)):
    return JSONResponse({"symbol": symbol.upper(), "expiries": expiries(symbol)})


@app.get("/api/chain")
async def api_chain(symbol: str, expiry: str | None = None,
                    email: str = Depends(require_user)):
    return JSONResponse(chain(symbol, expiry))


@app.get("/api/levels")
async def api_levels(symbol: str, email: str = Depends(require_user)):
    from engine.alpha_signal import compute_signal
    from storage.live_journal import record
    sig = compute_signal(symbol)
    record(sig)          # journal what we said, so it can be scored later
    return JSONResponse(sig)


@app.get("/api/journal")
async def api_journal(interval: int = 5, limit: int = 200,
                      email: str = Depends(require_user)):
    """Every ENTER call we made, scored against what price did next."""
    from engine.signal_review import review
    data = review(interval=interval, limit=limit)
    rows = sorted(data["rows"], key=lambda r: str(r.get("bar_time")), reverse=True)
    return JSONResponse({
        "summary": data["summary"],
        "logged": data["logged"],
        "rows": [{k: _clean(v) for k, v in r.items()} for r in rows[:limit]],
    })


def _clean(v):
    """NaN is not JSON. Null is."""
    try:
        import math
        if isinstance(v, float) and math.isnan(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


@app.get("/api/chart")
async def api_chart(symbol: str, days: int = 2, interval: int = 5,
                    email: str = Depends(require_user)):
    """Candlestick + EMA data for the live chart."""
    from data.live_feed import get_bars
    from engine.indicators import add_indicators
    feed = get_bars(symbol, days=days, interval=interval)
    if not feed.get("ok"):
        return JSONResponse({"ok": False, "error": feed.get("error")})
    df = add_indicators(feed["bars"]).tail(120)
    candles = [
        {"t": str(r["timestamp"])[11:16], "o": round(float(r["open"]), 2),
         "h": round(float(r["high"]), 2), "l": round(float(r["low"]), 2),
         "c": round(float(r["close"]), 2),
         "e9": None if _nan(r["ema9"]) else round(float(r["ema9"]), 2),
         "e15": None if _nan(r["ema15"]) else round(float(r["ema15"]), 2)}
        for _, r in df.iterrows()
    ]
    return JSONResponse({"ok": True, "symbol": symbol.upper(), "ltp": feed["ltp"],
                         "is_live": feed["is_live"], "source": feed["source"],
                         "candles": candles})


def _nan(v):
    try:
        import math
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


# ---------------------------------------------------------------- background sweep
async def _sweep_loop() -> None:
    """Journal ENTER signals across the universe while the app is up.

    Runs in-process so `uvicorn web.app:app` is the only command you need. The
    sweep is blocking (network + pandas), so it goes to a worker thread rather
    than stalling the event loop and with it every request the dashboard makes.
    """
    import asyncio
    import logging

    from engine.sweeper import DEFAULT_LIMIT, sweep

    logger = logging.getLogger("atlas.sweep")
    every = int(os.getenv("ATLAS_SWEEP_MINUTES", "5")) * 60
    limit = int(os.getenv("ATLAS_SWEEP_LIMIT", str(DEFAULT_LIMIT)))
    logger.info("Background sweep on: top %d every %dm "
                "(ATLAS_SWEEP=0 to disable)", limit, every // 60)
    while True:
        try:
            await asyncio.to_thread(sweep, limit)
        except asyncio.CancelledError:
            raise
        except Exception as e:                  # noqa: BLE001 — the loop must survive
            logger.warning("Sweep failed: %s", e)
        await asyncio.sleep(every)


@app.on_event("startup")
async def _start_sweep() -> None:
    if os.getenv("ATLAS_SWEEP", "1") != "1":
        return
    import asyncio
    app.state.sweep_task = asyncio.create_task(_sweep_loop())


@app.on_event("shutdown")
async def _stop_sweep() -> None:
    task = getattr(app.state, "sweep_task", None)
    if task:
        task.cancel()


@app.on_event("startup")
async def _warn_if_open() -> None:
    """Say it out loud at boot, not only when a login fails."""
    import logging
    if auth.bound_publicly() and auth.login_is_open():
        logging.getLogger("atlas").warning(
            "Atlas is bound to a non-loopback address with no SMTP and no "
            "ATLAS_ALLOWED_EMAILS — logins are blocked. %s", OPEN_LOGIN_ERROR)


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("ATLAS_BIND_HOST", "127.0.0.1")
    uvicorn.run("web.app:app", host=host, port=int(os.getenv("ATLAS_PORT", "8000")),
                reload=False)

"""Atlas web app — FastAPI backend with user login and the NSE screener.

Run:  uvicorn web.app:app --reload --port 8000
      (or: python -m web.app)
"""
from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web import auth
from web.data_service import chain, expiries, get_screen, refresh

BASE = Path(__file__).parent
app = FastAPI(title="Atlas")
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
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    if auth.authenticate(email, password):
        resp = RedirectResponse("/dashboard", status_code=303)
        resp.set_cookie(auth.COOKIE_NAME, auth.make_session(email),
                        max_age=auth.SESSION_MAX_AGE, httponly=True, samesite="lax")
        return resp
    return templates.TemplateResponse(request, "login.html", {"error": "Invalid email or password."})


@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse(request, "signup.html", {"error": None})


@app.post("/signup")
async def signup(request: Request, name: str = Form(""), email: str = Form(...),
                 password: str = Form(...)):
    ok, msg = auth.create_user(email, password, name)
    if not ok:
        return templates.TemplateResponse(request, "signup.html", {"error": msg})
    resp = RedirectResponse("/dashboard", status_code=303)
    resp.set_cookie(auth.COOKIE_NAME, auth.make_session(email),
                    max_age=auth.SESSION_MAX_AGE, httponly=True, samesite="lax")
    return resp


@app.get("/logout")
async def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(auth.COOKIE_NAME)
    return resp


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, email: str = Depends(require_user)):
    user = auth.get_user(email) or {"name": email}
    return templates.TemplateResponse(request, "dashboard.html", {"user": user})


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
    from engine.levels import compute_levels
    return JSONResponse(compute_levels(symbol))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=False)

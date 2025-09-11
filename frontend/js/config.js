(function (w) {
    try {
        var url = new URL(window.location.href);
        var apiHost = url.searchParams.get("api_host") || window.location.hostname;
        var apiPort = url.searchParams.get("api_port") || "5000";
        w.API_BASE = "http://" + apiHost + ":" + apiPort;
    } catch (e) {
        w.API_BASE = "http://localhost:5000";
    }
})(window);





import React, { useState } from "react";
import ReactDOM from "react-dom";
import ReactMarkdown from "react-markdown";
import MarkNav from "markdown-navbar";
import article from "./article";

import "markdown-navbar/dist/navbar.css";
import "./styles.css";

function App() {
  const [navVisible, setNavVisible] = useState(true);

  return (
    <div className="App">
      <div className="article-container">
        <ReactMarkdown source={article} />
      </div>
      <div className={`nav-container ${navVisible ? "show" : "hide"}`}>
        <div
          className="toggle-btn"
          onClick={() => {
            setNavVisible(!navVisible);
          }}
        >
          {navVisible ? "MENU →" : "← MENU"}
        </div>
        <MarkNav source={article} />
      </div>
    </div>
  );
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
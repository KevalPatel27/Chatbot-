import { useState, useRef, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import "./Chatbox.css";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faThumbsDown, faThumbsUp } from "@fortawesome/free-regular-svg-icons";

const ChatBox = () => {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState([]);
  const [showForm, setShowForm] = useState(false);
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    user_issue: "",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, showForm]);

  const handleSend = async () => {
    if (!prompt.trim()) return;

    const userMessage = { role: "user", content: prompt };
    setMessages((prev) => [...prev, userMessage]);
    try {
      const res = await axios.post("http://127.0.0.1:5000/ask", {
        prompt,
      });
      const botMessage = { role: "bot", content: res.data.response };
      setMessages((prev) => [...prev, botMessage]);
      setPrompt("");
    } catch (err) {
      console.error("Error:", err);
      setMessages((prev) => [
        ...prev,
        { role: "bot", content: "❌ Error occurred" },
      ]);
    }
  };

  const isClicked = () => {
    setMessages((prev) => [
      ...prev,
      {
        role: "bot",
        content: "Sorry to hear that. Please fill out the support form below.",
      },
    ]);
    setShowForm(true);
    setPrompt("");
    return;
  };

  const handleFormSubmit = async (e) => {
    setIsSubmitting(true);
    e.preventDefault();
    try {
      await axios.post("http://127.0.0.1:5000/send-support-email", formData);

      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          content:
            "✅ Your request has been submitted. Our support team will contact you soon.",
        },
      ]);
      setShowForm(false);
      setIsSubmitting(false);
      setFormData({ name: "", email: "", user_issue: "" });
    } catch (error) {
      console.error("Form submission error:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          content: "❌ Failed to send support request. Please try again.",
        },
      ]);
    }
  };
  const handleClick = () => {
    setShowForm(false);
    setMessages((prev) => [
      ...prev,
      {
        role: "bot",
        content:
          "❌ Failed to send support request.You have not filled the form.",
      },
    ]);
  };
  return (
    <div className="chatbox ">
      <div
        className="messages"
        style={showForm ? { height: "calc(100vh - 443px)" } : {}}
      >
        <div className="bot">
          <strong>{"🤖 Bot"}:</strong>
          <p>Hi there! How can I help you today?</p>
        </div>
        {messages.map((msg, idx) => (
          <div key={idx} className={msg.role}>
            <strong>{msg.role === "user" ? "🧠 You" : "🤖 Bot"}:</strong>
            <ReactMarkdown>{msg.content}</ReactMarkdown>
            {msg.role == "bot" && !showForm && (
              <div className="icons">
                <FontAwesomeIcon
                  icon={faThumbsUp}
                  size="lg"
                  className="thumbColor thumbup"
                />
                <FontAwesomeIcon
                  icon={faThumbsDown}
                  size="lg"
                  className="thumbColor thumbdown"
                  onClick={isClicked}
                />
              </div>
            )}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {showForm ? (
        <form className="support-form" onSubmit={handleFormSubmit}>
          <h3>Support Form</h3>
          <input
            className="textarea"
            type="text"
            placeholder="Your Name"
            value={formData.name}
            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
            required
          />
          <input
            type="email"
            placeholder="Your Email"
            value={formData.email}
            onChange={(e) =>
              setFormData({ ...formData, email: e.target.value })
            }
            pattern="[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$"
            required
          />
          <textarea
            placeholder="Describe your issue..."
            value={formData.user_issue}
            onChange={(e) =>
              setFormData({ ...formData, user_issue: e.target.value })
            }
            required
          />
          <button type="submit" disabled={isSubmitting}>
            {isSubmitting ? "Submitting..." : "Submit"}
          </button>
          <button type="button" className="cancelBtn" onClick={handleClick}>
            cancel
          </button>
        </form>
      ) : (
        <div className="input-section">
          <input
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder="Type your message..."
          />
          <button onClick={handleSend}>Send</button>
        </div>
      )}
    </div>
  );
};

export default ChatBox;

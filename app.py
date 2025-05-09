from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import pymysql
from groq import Groq
from chroma_memory import store_memory, retrieve_memory
from mail_service import send_email

load_dotenv()

app = Flask(__name__)
# Allow CORS only for your frontend URL
CORS(app, origins=["http://localhost:5173"])

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def connect_to_db():
    try:
        connection = pymysql.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            port=int(os.getenv("DB_PORT"))
        )
        print("✅ Connected to MySQL.")
        return connection
    except pymysql.MySQLError as e:
        print("❌ DB connection error:", e)
        return None


def table_schema_getter(db_connection,tables=["wp_posts", "wp_postmeta", "wp_users", "wp_commentmeta","wp_comments","wp_terms","wp_termmeta"]):
    try:
        cursor = db_connection.cursor()
        schema_info = ""
        for table in tables:
            schema_info += f"\nTable:{table}\n"
            cursor.execute(f"DESCRIBE {table}")
            columns = cursor.fetchall()
            for column in columns:
                field, col_type, is_nullable, key = column[0], column[1], column[2], column[3]
                schema_info += f"- {field} ({col_type}, Nullable: {is_nullable}, Key: {key})\n"
        return schema_info.strip()
    except Exception as e:
        return f"Error retrieving schema: {e}"
user_prompt=""
@app.route("/ask", methods=["POST"])
def ask_bot():
    data = request.json
    user_prompt = data.get("prompt")
    if not user_prompt:
        return jsonify({"error": "No prompt provided"}), 400

    db_connection = None
    
    try:
        print("🔍 Step 0: Searching memory...")
        similar_interactions = retrieve_memory(query=user_prompt, top_k=3)
        context_from_memory = "\n\nRelevant past interactions:\n" + "\n---\n".join(
            [m["document"] for m in similar_interactions]
        ) if similar_interactions else ""
        
        print(f"Found {len(similar_interactions)} similar interactions")

        db_connection = connect_to_db()
        if db_connection is None:
            return jsonify({"error": "Failed to connect to the database"}), 500

        print("😎 Step 1: restructed the prompt...")
        re_prompt = f"""
        You are an AI assistant helping generate precise SQL queries based on user intent.

        The original user question may be vague, incomplete, or unclear. Your task is to rephrase or clarify it into a more direct and unambiguous format, specifically optimized for generating SQL SELECT queries only.

        Here is the user's original question:
        {user_prompt}

        Instructions:
        - Rephrase the question in a clearer and more specific way.
        - Focus on what the user *wants to know or retrieve* from the database.
        - Do NOT allow any form of insertion, updating, deletion, dropping, or truncating data.
        - Strictly ensure that the clarified instruction is suitable only for SELECT-based queries.
        - Keep it as a **single-line query-style** instruction. No extra commentary.
        - Do NOT include SQL — just the improved prompt.

        Respond with only the clarified prompt.
        """

        completion = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": re_prompt}],
            temperature=0,
            max_tokens=512,
            top_p=1
        )
        user_prompt1 = completion.choices[0].message.content.strip()
        print(user_prompt1)
        
        #getting tabel schema
        table_schema = table_schema_getter(db_connection)
        
        
        print("🧠 Step 2: Generating SQL...")
        prompt_for_model = f"""Given the schema of the tables, generate a single SELECT SQL query to retrieve data.
        Schema: {table_schema}
        User Question: {user_prompt1}
        Context: {context_from_memory}
        - Focus on what the user wants to retrieve from the database.
        - Do not perform any INSERT, UPDATE, DELETE, DROP, or TRUNCATE operations.
        - Only provide one SQL query not multiple SQl query only one. No additional text. No explanations or comments."""
        completion = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt_for_model}],
            temperature=0,
            max_tokens=512,
            top_p=1
        )
        sql_query = completion.choices[0].message.content.strip()
        cleaned_sql = sql_query.replace("```sql", "").replace("```", "").strip()
        print("✅ Generated SQL:", cleaned_sql)
        cursor = db_connection.cursor()
        cursor.execute(cleaned_sql)
        result = cursor.fetchall()
        # print(result)
        
        # Convert to readable list of dicts
        columns = [desc[0] for desc in cursor.description]
        rows = [dict(zip(columns, row)) for row in result]

        print("💬 Step 3: Generating final response...")
        user_prompt_with_db_answer = f"""
            You are a helpful, ethical, and polite customer support assistant for our company. Your sole responsibility is to assist users based ONLY on the information retrieved from the database and relevant past conversation memory.

            Context:
            - User's Question: {user_prompt}
            - Retrieved Information from Database: {rows}
            - Previous Conversation Memory: {context_from_memory}

            Instructions:
            1. **Only respond if you find directly relevant and verifiable information in the retrieved Database Results.**
            2. **If the user’s query is unrelated to company services or no relevant data is found**, respond respectfully with:
            "I'm sorry, I can only assist based on available service records. Please rephrase your question."
            3. **Never display, reference, or imply any technical database details**, including:
            - SQL queries
            - Table or column names
            - Schema or system architecture
            4. You must **never fabricate, assume, or guess answers**. Respond only based on the verified data available.
            5. You have **read-only access** — never suggest, imply, or perform creation, updates, or deletions.
            6. Always communicate in a **respectful, unbiased, and empathetic** tone.
            7. **Do not make judgments, express personal opinions, or suggest actions outside the company’s available services**.
            8. Every message must end with the following line in bold:
            
            **if you found this response helpful, please click the button below to let us know!**
            
            Ethical Guidelines:
            - Maintain full confidentiality of user data and internal information.
            - Treat every user equally, without bias or discrimination.
            - Ensure transparency by only sharing information that is verifiable through company records.
            - Avoid overpromising or providing unsupported guidance.
            - Do not attempt to manipulate or persuade — your role is to inform and assist.

            Assistant Notes:
            - Do not mention or refer to the existence of "database", "memory", or "retrieval systems".
            - Respond naturally and professionally, as if you are a knowledgeable human agent.
            - Never express or imply that you are an AI or are accessing backend systems.
        """



        final_completion = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": user_prompt_with_db_answer}],
            temperature=0.7,
            max_tokens=1024,
            top_p=1
        )

        reply_md = final_completion.choices[0].message.content

        print("🧠 Storing interaction to memory...")
        store_memory(
            prompt=user_prompt,
            sql=sql_query,
            result=str(rows),
            final_response=reply_md
        )

        return jsonify({"response": reply_md})

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)}), 500

    finally:
        if db_connection:
            db_connection.close()

#Handing form-fill request and sending Email
@app.route('/send-support-email', methods=['POST'])
def send_support_email():
    data = request.get_json()
    print(data)
    name = data.get('name')
    email = data.get('email')
    issue = data.get('prompt')
    send_email(name,email,issue)  # Your mail_service.py function
    return jsonify({"status": "success"}), 200


if __name__ == "__main__":
    app.run(debug=True)

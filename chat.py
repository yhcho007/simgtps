import socket
import threading
import os
import re
from typing import Dict, List, Set, Tuple

# --- Configuration ---
HOST = '0.0.0.0'
PORT = 8126
DOWNLOAD_DIR = 'downloads'

# --- Global Variables ---
clients: Dict[str, socket.socket] = {}
client_addresses: Dict[Tuple[str, int], str] = {}
rejected_users: Set[str] = set()
nickname: str | None = None


# --- File Transfer Helper ---
def handle_file_transfer(conn: socket.socket, file_path: str):
    file_size = os.path.getsize(file_path)
    file_name = os.path.basename(file_path)
    conn.send(f"FILE_TRANSFER_START:{file_name}:{file_size}".encode('utf-8'))
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(1024)
            if not chunk:
                break
            conn.sendall(chunk)
    print(f"[File] '{file_name}' sent successfully.")


def receive_file_transfer(conn: socket.socket, file_info: str):
    try:
        _, file_name, file_size_str = file_info.split(':', 2)
        file_size = int(file_size_str)
        if not os.path.exists(DOWNLOAD_DIR):
            os.makedirs(DOWNLOAD_DIR)
        file_path = os.path.join(DOWNLOAD_DIR, file_name)
        with open(file_path, 'wb') as f:
            bytes_received = 0
            while bytes_received < file_size:
                chunk = conn.recv(1024)
                if not chunk:
                    break
                f.write(chunk)
                bytes_received += len(chunk)
        print(f"[File] '{file_name}' ({file_size} bytes) received successfully.")
    except Exception as e:
        print(f"[Error] Failed to receive file: {e}")


# --- Command Handler ---
def handle_commands(command: str):
    global nickname
    parts = command.split(' ', 2)
    cmd = parts[0].lower()

    # 🚩 quit 명령어 추가
    if cmd == 'quit':
        return 'QUIT'

    if cmd == 'list':
        user_list = ', '.join(clients.keys()) if clients else 'No users online.'
        print(f"Online users: {user_list}")

    elif cmd == 'iam':
        if len(parts) > 1:
            new_nickname = parts[1]
            if nickname and nickname in clients:
                del clients[nickname]
                print(f"You have changed your nickname to '{new_nickname}'.")
            nickname = new_nickname
            clients[nickname] = s
            print(f"You are now '{nickname}'.")
        else:
            print("Usage: iam [your_nickname]")

    elif cmd == 'to':
        if len(parts) > 2:
            recipient = parts[1]
            message_content = parts[2]
            if recipient in rejected_users:
                print(f"[Rejected] You have rejected messages from '{recipient}'.")
                return
            if recipient in clients:
                if nickname in rejected_users:
                    clients[nickname].send(f"REJECTED:{recipient}".encode('utf-8'))
                    return
                formatted_message = f"CHAT_MESSAGE:{nickname}:{message_content}"
                clients[recipient].send(formatted_message.encode('utf-8'))
                print(f"-> Message sent to '{recipient}'.")
            else:
                print(f"[Error] User '{recipient}' is not online.")
        else:
            print("Usage: to [recipient_nickname] [message]")

    elif cmd == 'file':
        if len(parts) > 2:
            recipient = parts[1]
            file_path = parts[2]
            if not os.path.exists(file_path):
                print(f"[Error] File not found: {file_path}")
                return
            if recipient in clients:
                try:
                    conn = clients[recipient]
                    handle_file_transfer(conn, file_path)
                except Exception as e:
                    print(f"[Error] File transfer failed: {e}")
            else:
                print(f"[Error] User '{recipient}' is not online.")
        else:
            print("Usage: file [recipient_nickname] [file_path]")

    elif cmd == 'reject':
        if len(parts) > 1:
            user_to_reject = parts[1]
            if user_to_reject not in rejected_users:
                rejected_users.add(user_to_reject)
                print(f"You will now reject all communication from '{user_to_reject}'.")
            else:
                print(f"You are already rejecting '{user_to_reject}'.")
        else:
            print("Usage: reject [user_nickname]")

    else:
        print("Unknown command. Type 'list' for a list of commands.")


# --- Server/Client Logic ---
def listen_for_messages(conn: socket.socket, addr: Tuple[str, int]):
    while True:
        try:
            data = conn.recv(1024).decode('utf-8')
            if not data:
                break
            if data.startswith("CHAT_MESSAGE:"):
                _, sender, message = data.split(':', 2)
                if sender in rejected_users:
                    print(f"\n[Rejected] Message from '{sender}' was rejected.")
                    conn.send("REJECTED".encode('utf-8'))
                else:
                    print(f"\n[Message from {sender}]: {message}")
            elif data.startswith("FILE_TRANSFER_START:"):
                if "REJECTED" in data:
                    print(f"\n[File Transfer] Your file transfer was rejected by the recipient.")
                    continue
                _, file_name, file_size_str = data.split(':', 2)
                if file_name in rejected_users:
                    print(f"\n[Rejected] File from '{file_name}' was rejected.")
                    continue
                print(f"\n[File Transfer] Receiving file '{file_name}'...")
                receive_file_transfer(conn, data)
            elif data.startswith("REJECTED:"):
                _, rejected_user = data.split(':', 1)
                print(f"\n[Rejected] Your message to '{rejected_user}' was rejected.")
            else:
                print(f"\n[Received from {addr}]: {data}")
        except Exception as e:
            print(f"[Connection Error] Disconnected from {addr}. Reason: {e}")
            conn.close()
            if addr in client_addresses:
                removed_nickname = client_addresses[addr]
                del clients[removed_nickname]
                del client_addresses[addr]
                print(f"User '{removed_nickname}' has left the chat.")
            break


# --- Main Program Loop ---
print("Simple P2P Chat and File Transfer Program")
print("Type 'list', 'iam [nickname]', 'to [user] [message]', 'file [user] [path]', 'reject [user]', or 'quit'.")

# Create a socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)

print(f"Listening on port {PORT}...")


def accept_connections():
    while True:
        try:
            conn, addr = s.accept()
            print(f"Connected by {addr}")
            client_addresses[addr] = "unknown"
            thread = threading.Thread(target=listen_for_messages, args=(conn, addr))
            thread.daemon = True
            thread.start()
        except OSError:  # 소켓 종료 시 발생하는 예외 처리
            break


accept_thread = threading.Thread(target=accept_connections)
accept_thread.daemon = True
accept_thread.start()

# 🚩 메인 루프 수정: 'quit' 명령어를 확인하고 종료
while True:
    try:
        command_input = input(">> ")
        if command_input.lower() == 'quit':
            print("프로그램을 종료합니다.")
            break
        handle_commands(command_input)
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
        break
    except Exception as e:
        print(f"[Error] {e}")

s.close()
print("프로그램이 정상적으로 종료되었습니다.")


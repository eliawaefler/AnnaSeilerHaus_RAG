css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://www.insel.ch/_ari/115280/49841742b8afbc44928918244fb4c6f9b487d5b3/9f6e35f65cbd0d6c47c145f90b1d5a297eb50bcd/1400/0/og/20230704-Anna-Seiler-Haus-009-screen.jpg.webp" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://media.licdn.com/dms/image/C4D03AQHi5rJfheyUtQ/profile-displayphoto-shrink_800_800/0/1638174649461?e=2147483647&v=beta&t=KOsttcLGIwB9pBEVfceHj-ckv_zPHs-2COyrp7aYR-k">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
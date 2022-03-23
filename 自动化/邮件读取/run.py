import poplib

import zmail
from tqdm import tqdm

start_date = input('请输入起始时间，精确到年月日时(格式如2022-03-18 13:00:00): ')
end_date = input('请输入终止时间，精确到年月日时(格式如2022-03-18 13:00:00): ')

# start_date = '2022-03-18 13:00:00'
# end_date = '2022-03-18 14:00:00'

username = ''  # 填你的邮箱
password = ''  # 填你的授权码

# 登录邮箱
server = poplib.POP3_SSL(host='pop.163.com')
server.user(username)
server.pass_(password)

# 获取所有邮件
num = server.stat()[0]
_range = range(1, num + 1)

mail_hdrs = []
for count in tqdm(_range, desc='收取邮件中'):
    mail = server.retr(count)[1]
    mail_hdrs.append(mail)

# 解析邮件
mails = []
for mail_header in tqdm(mail_hdrs, desc='解析邮件中'):
    parsed_mail = zmail.parser.parse(mail_header)
    mails.append(parsed_mail)

# 筛选今天的邮件
require_mails = []

start_date_str = start_date + '+08:00'
end_date_str = end_date + '+08:00'

print(start_date, end_date)

for mail in tqdm(mails, desc='筛选邮件中'):
    mail_date = mail.get('date')
    mail_date = str(mail_date)

    if mail_date >= start_date_str and mail_date <= end_date_str:
        require_mails.append(mail)

for mail in require_mails:
    mail_date = mail.get('date')
    mail_date = str(mail_date).split('+')[0]

    mail_from = mail.get('From')
    subject = mail.get('Subject')
    content = mail.get('content_text') or mail.get('content_html')

    print('{}，{}发来{}，邮件内容是：{}'.format(mail_date, mail_from, subject,
                                      content[0]))
    print(
        '==============================================================================================='
    )

if len(require_mails) == 0:
    print('{}-{}，暂无邮件处理'.format(start_date, end_date))

print('{}-{} 共收到{}封邮件'.format(start_date, end_date, len(require_mails)))

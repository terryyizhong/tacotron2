""" from https://github.com/keithito/tacotron """
import re
from text import cleaners
from text.symbols import symbols, symbols_chs



# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def text_to_sequence(text, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []


  # Mappings from symbol to numeric ID and vice versa:
  _symbol_to_id = {s: i for i, s in enumerate(symbols)}
  _id_to_symbol = {i: s for i, s in enumerate(symbols)}
  if cleaner_names[0] == 'chinese_cleaners':

    _symbol_to_id = {s: i for i, s in enumerate(symbols_chs)}
    _id_to_symbol = {i: s for i, s in enumerate(symbols_chs)}

  # Check for curly braces and treat their contents as ARPAbet:
  while len(text):
    m = _curly_re.match(text)
    if not m:
      sequence += _symbols_to_sequence(_clean_text(text, cleaner_names), _symbol_to_id)
      break
    sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
    sequence += _arpabet_to_sequence(m.group(2))
    text = m.group(3)

  return sequence


#def sequence_to_text(sequence):
#  '''Converts a sequence of IDs back to a string'''
#  result = ''
#  for symbol_id in sequence:
#    if symbol_id in _id_to_symbol:
#      s = _id_to_symbol[symbol_id]
#      # Enclose ARPAbet back in curly braces:
#      if len(s) > 1 and s[0] == '@':
#        s = '{%s}' % s[1:]
#      result += s
#  return result.replace('}{', ' ')

def an_to_cn(text):
    '''convert integer to Chinese numeral'''
    an_list = re.findall('(\d+)',text)
    if len(an_list) == 0:
        return text
    for an in an_list:
        text = text.replace(an,_to_cn(an),1)
    return text


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(symbols, _symbol_to_id):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s, _symbol_to_id)]


def _arpabet_to_sequence(text, _symbols_to_id):
  return _symbols_to_sequence(['@' + s for s in text.split()], _symbol_to_id)


def _should_keep_symbol(s, _symbol_to_id):
  return s in _symbol_to_id and s is not '_' and s is not '~'





def _to_cn(number):
    """ convert integer to Chinese numeral """

    chinese_numeral_dict = {
        '0': '零',
        '1': '一',
        '2': '二',
        '3': '三',
        '4': '四',
        '5': '五',
        '6': '六',
        '7': '七',
        '8': '八',
        '9': '九'
    }
    chinese_unit_map = [('', '十', '百', '千'),
                        ('万', '十万', '百万', '千万'),
                        ('亿', '十亿', '百亿', '千亿'),
                        ('兆', '十兆', '百兆', '千兆'),
                        ('吉', '十吉', '百吉', '千吉')]
    chinese_unit_sep = ['万', '亿', '兆', '吉']

    reversed_n_string = reversed(str(number))

    result_lst = []
    unit = 0

    for integer in reversed_n_string:
        if integer is not '0':
            result_lst.append(chinese_unit_map[unit // 4][unit % 4])
            result_lst.append(chinese_numeral_dict[integer])
            unit += 1
        else:
            if result_lst and result_lst[-1] != '零':
                result_lst.append('零')
            unit += 1

    result_lst.reverse()

    # clean convert result, make it more natural
    if result_lst[-1] is '零':
        result_lst.pop()

    result_lst = list(''.join(result_lst))

    for unit_sep in chinese_unit_sep:
        flag = result_lst.count(unit_sep)
        while flag > 1:
            result_lst.pop(result_lst.index(unit_sep))
            flag -= 1

    '''
    length = len(str(number))
    if 4 < length <= 8:
        flag = result_lst.count('万')
        while flag > 1:
            result_lst.pop(result_lst.index('万'))
            flag -= 1
    elif 8 < length <= 12:
        flag = result_lst.count('亿')
        while flag > 1:
            result_lst.pop(result_lst.index('亿'))
            flag -= 1
    elif 12 < length <= 16:
        flag = result_lst.count('兆')
        while flag > 1:
            result_lst.pop(result_lst.index('兆'))
            flag -= 1
    elif 16 < length <= 20:
        flag = result_lst.count('吉')
        while flag > 1:
            result_lst.pop(result_lst.index('吉'))
            flag -= 1
    '''

    return ''.join(result_lst)

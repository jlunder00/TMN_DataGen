# Authored by: Jason Lunder, EWUID: 01032294, Github: https://github.com/jlunder00/

# TMN_DataGen/TMN_DataGen/utils/tokenizers.py
from abc import ABC, abstractmethod
import re
import stanza
from typing import List

class BaseTokenizer(ABC):
    def __init__(self, config, vocabs=None, logger=None):
        """
        Args:
            config: Configuration object.
            vocab (set): A set of valid tokens (typically from your word2vec model).
        """
        self.config = config
        self.logger = logger
        # Store vocab as a set for quick membership testing.
        self.vocabs = vocabs if vocabs is not None else [set()]
        self.allowed_two_letter = {
            "am", "an", "as", "at", "ax",
            "be", "by",
            "do",
            "go",
            "he", "hi",
            "if", "in", "is", "it",
            "me", "my",
            "no",
            "of", "oh", "on", "or", "ok", "ox",
            "pi",
            "so",
            "to",
            "up",
            "us",
            "we"
        }
        self.allowed_one_letter = {'i', 'a'}
        # If vocabs is provided, filter each one.
        if vocabs is not None:
            filtered_vocabs = []
            for vocab in vocabs:
                filtered_vocab = {
                    word for word in vocab 
                    if len(word) > 2 or 
                    (len(word) == 2 and word in self.allowed_two_letter) or 
                    (len(word) == 1 and word in self.allowed_one_letter)
                }
                filtered_vocabs.append(filtered_vocab)
            self.vocabs = filtered_vocabs
        else:
            # Default to an empty set if not provided.
            self.vocabs = [set()]

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass

    def tokenize_with_vocab(self, text: str) -> List[str]:
        tokens = self.tokenize(text)
        processed_tokens = []
        for token in tokens:
            if any(token in v for v in self.vocabs) or self.check_number(token) or len(token) < 5: #if less than 5 chars, highly unlikely to be a smashed word
                #the word is valid
                processed_tokens.append(token)
            else:
                valid_tokens = self.subword_tokenize(token)
                if valid_tokens:
                    processed_tokens.extend(valid_tokens)
                else:
                    self.logger.debug(f"Dropping token: {token}, not found in vocab and no subtokens could be found in vocab")

        return processed_tokens

    def check_number(self, t):
        return t.replace('.','',1).isdigit()

    def clean_token(self, token: str) -> str:
        """
        Aggressively clean the token by removing digits.
        (You could extend this to remove other extraneous characters as needed.)
        """
        return re.sub(r'\d+', '', token)

    def subword_tokenize(self, token: str) -> List[str]:
        """
        Attempts to segment a token into valid subwords using dynamic programming.
        First, try the DP approach on the original token.
        If that fails, clean the token (e.g. remove digits) and try DP again.
        If still no complete segmentation is found, fall back to the greedy method
        using the original token.
        """
        def dp_segment(tkn: str) -> List[str]:
            n_local = len(tkn)
            dp = [None] * (n_local + 1)
            dp[0] = []  # Base case: empty segmentation

            # Determine maximum word length from any vocab.
            max_word_len_local = max((len(word) for vocab in self.vocabs for word in vocab), default=n_local)

            def candidate_key(segmentation: List[str]):
                counts = [sum(1 for vocab in self.vocabs if seg in vocab) for seg in segmentation]
                return (len(segmentation), tuple(-c for c in counts), tuple(-len(seg) for seg in segmentation))

            for i in range(1, n_local + 1):
                for j in range(max(0, i - max_word_len_local), i):
                    segment = tkn[j:i]
                    if dp[j] is not None and self.is_valid_segment(segment):
                        candidate = dp[j] + [segment]
                        if dp[i] is None or candidate_key(candidate) < candidate_key(dp[i]):
                            dp[i] = candidate
            return dp[n_local]

        # First, try DP on the original token.
        segmentation = dp_segment(token)
        if segmentation is not None:
            return segmentation
        # If DP on the original token fails, clean the token and try again.
        cleaned_token = self.clean_token(token)
        if cleaned_token != token:
            segmentation_clean = dp_segment(cleaned_token)
            if segmentation_clean is not None:
                return segmentation_clean
        # If still unsuccessful, fall back to the greedy method using the original token.
        return self.greedy_subword_tokenize(token)




    def is_valid_segment(self, segment: str) -> bool:
        """
        A segment is valid if it appears in any of the filtered vocabs.
        """
        return any(segment in vocab for vocab in self.vocabs)

    def greedy_subword_tokenize(self, token: str) -> List[str]:
        """
        A more efficient greedy approach:
          - For each possible substring length, from n (the full token length) down to 1,
            gather all valid substrings of that length.
          - If any valid candidates are found at that length, choose the best one:
              * First, by the highest frequency (number of vocab sets in which it appears),
              * Then, if tied, by the earliest (lowest) start index.
          - Let left_remainder = token[0:best_start] and right_remainder = token[best_end:].
          - Include a remainder if its length is less than 4 (assuming it's an abbreviation).
          - Return the list: [left_remainder (if included), best segment, right_remainder (if included)].
          - If no valid substring is found and the token is shorter than 4 letters, return the token;
            otherwise, return an empty list.
        """
        n = len(token)
        # Try each possible substring length, starting from the full token length.
        for seg_length in range(n, 0, -1):
            candidates = []
            # For each possible starting index for a substring of length seg_length.
            for i in range(n - seg_length + 1):
                j = i + seg_length
                segment = token[i:j]
                if self.is_valid_segment(segment):
                    freq = sum(1 for vocab in self.vocabs if segment in vocab)
                    candidates.append((i, j, segment, freq))
            if candidates:
                # Among candidates of this length, select the one with the highest frequency.
                # If there's a tie, choose the one that appears earliest (lowest start index).
                best = max(candidates, key=lambda x: (x[3], -x[0]))
                best_start, best_end, best_seg, _ = best
                result = []
                left_remainder = token[:best_start]
                right_remainder = token[best_end:]
                total_remainder_length = len(left_remainder) + len(right_remainder)
                total_length = total_remainder_length + len(best_seg)
                if total_length > 8 and total_remainder_length > 5 and len(best_seg) > 3: #longer words are likely technical
                    result = [token] #take whole token
                else:
                    if left_remainder and len(left_remainder) < 4:
                        result.append(left_remainder)
                    result.append(best_seg)
                    if right_remainder and len(right_remainder) < 4:
                        result.append(right_remainder)
                return result
        # If no valid substring is found:
        return [token] if n < 4 else []




class RegexTokenizer(BaseTokenizer):
    def __init__(self, config, vocab=None, logger=None):
        super().__init__(config, vocab, logger)
        self.min_len = config.preprocessing.min_token_length
        self.max_len = config.preprocessing.max_token_length
        
    def tokenize(self, text: str) -> List[str]:
        # Simple word boundary tokenization
        tokens = re.findall(r'\b\w+\b', text)
        # Apply length filters
        tokens = [t for t in tokens 
                 if self.min_len <= len(t) <= self.max_len]
        return tokens

class StanzaTokenizer(BaseTokenizer):
    def __init__(self, config, vocab=None, logger=None):
        super().__init__(config, vocab, logger)
        try:
            self.nlp = stanza.Pipeline(
                lang=config.preprocessing.language,
                processors='tokenize',
                use_gpu=True,
                verbose=False
            )
        except Exception as e:
            raise ValueError(f"Failed to load Stanza: {e}")
            
    def tokenize(self, text: str) -> List[str]:
        doc = self.nlp(text)
        tokens = [word.text for sent in doc.sentences 
                 for word in sent.words]
        return tokens

## Prompts for Evaluation Metrics

### Hallucination (https://www.comet.com/docs/opik/evaluation/metrics/hallucination)

```
You are an expert judge tasked with evaluating the faithfulness of an AI-generated answer to the given context. Analyze the provided INPUT, CONTEXT, and OUTPUT to determine if the OUTPUT contains any hallucinations or unfaithful information.

Guidelines:

1. The OUTPUT must not introduce new information beyond what's provided in the CONTEXT.
2. The OUTPUT must not contradict any information given in the CONTEXT.
3. The OUTPUT should not contradict well-established facts or general knowledge.
4. Ignore the INPUT when evaluating faithfulness; it's provided for context only.
5. Consider partial hallucinations where some information is correct but other parts are not.
6. Pay close attention to the subject of statements. Ensure that attributes, actions, or dates are correctly associated with the right entities (e.g., a person vs. a TV show they star in).
7. Be vigilant for subtle misattributions or conflations of information, even if the date or other details are correct.
8. Check that the OUTPUT doesn't oversimplify or generalize information in a way that changes its meaning or accuracy.

Analyze the text thoroughly and assign a hallucination score between 0 and 1, where:

- 0.0: The OUTPUT is entirely faithful to the CONTEXT
- 1.0: The OUTPUT is entirely unfaithful to the CONTEXT

{examples_str}

INPUT (for context only, not to be used for faithfulness evaluation):
{input}

CONTEXT:
{context}

OUTPUT:
{output}

It is crucial that you provide your answer in the following JSON format:
{{
    "score": <your score between 0.0 and 1.0>,
    "reason": ["reason 1", "reason 2"]
}}
Reasons amount is not restricted. Output must be JSON format only.
```

### Usefulness (https://www.comet.com/docs/opik/evaluation/metrics/usefulness)
```
You are an impartial judge tasked with evaluating the quality and usefulness of AI-generated responses.

Your evaluation should consider the following key factors:
- Helpfulness: How well does it solve the user's problem?
- Relevance: How well does it address the specific question?
- Accuracy: Is the information correct and reliable?
- Depth: Does it provide sufficient detail and explanation?
- Creativity: Does it offer innovative or insightful perspectives when appropriate?
- Level of detail: Is the amount of detail appropriate for the question?

###EVALUATION PROCESS###

1. **ANALYZE** the user's question and the AI's response carefully
2. **EVALUATE** how well the response meets each of the criteria above
3. **CONSIDER** the overall effectiveness and usefulness of the response
4. **PROVIDE** a clear, objective explanation for your evaluation
5. **SCORE** the response on a scale from 0.0 to 1.0:
   - 1.0: Exceptional response that excels in all criteria
   - 0.8: Excellent response with minor room for improvement
   - 0.6: Good response that adequately addresses the question
   - 0.4: Fair response with significant room for improvement
   - 0.2: Poor response that barely addresses the question
   - 0.0: Completely inadequate or irrelevant response

###OUTPUT FORMAT###

Your evaluation must be provided as a JSON object with exactly two fields:
- "score": A float between 0.0 and 1.0
- "reason": A brief, objective explanation justifying your score based on the criteria above

Now, please evaluate the following:

User Question: {input}
AI Response: {output}

Provide your evaluation in the specified JSON format.
```

### Answer relevance (https://www.comet.com/docs/opik/evaluation/metrics/answer_relevance)
```
YOU ARE AN EXPERT IN NLP EVALUATION METRICS, SPECIALLY TRAINED TO ASSESS ANSWER RELEVANCE IN RESPONSES
PROVIDED BY LANGUAGE MODELS. YOUR TASK IS TO EVALUATE THE RELEVANCE OF A GIVEN ANSWER FROM
ANOTHER LLM BASED ON THE USER'S INPUT AND CONTEXT PROVIDED.

###INSTRUCTIONS###

- YOU MUST ANALYZE THE GIVEN CONTEXT AND USER INPUT TO DETERMINE THE MOST RELEVANT RESPONSE.
- EVALUATE THE ANSWER FROM THE OTHER LLM BASED ON ITS ALIGNMENT WITH THE USER'S QUERY AND THE CONTEXT.
- ASSIGN A RELEVANCE SCORE BETWEEN 0.0 (COMPLETELY IRRELEVANT) AND 1.0 (HIGHLY RELEVANT).
- RETURN THE RESULT AS A JSON OBJECT, INCLUDING THE SCORE AND A BRIEF EXPLANATION OF THE RATING.

###CHAIN OF THOUGHTS###

1. **Understanding the Context and Input:**
   1.1. READ AND COMPREHEND THE CONTEXT PROVIDED.
   1.2. IDENTIFY THE KEY POINTS OR QUESTIONS IN THE USER'S INPUT THAT THE ANSWER SHOULD ADDRESS.

2. **Evaluating the Answer:**
   2.1. COMPARE THE CONTENT OF THE ANSWER TO THE CONTEXT AND USER INPUT.
   2.2. DETERMINE WHETHER THE ANSWER DIRECTLY ADDRESSES THE USER'S QUERY OR PROVIDES RELEVANT INFORMATION.
   2.3. CONSIDER ANY EXTRANEOUS OR OFF-TOPIC INFORMATION THAT MAY DECREASE RELEVANCE.

3. **Assigning a Relevance Score:**
   3.1. ASSIGN A SCORE BASED ON HOW WELL THE ANSWER MATCHES THE USER'S NEEDS AND CONTEXT.
   3.2. JUSTIFY THE SCORE WITH A BRIEF EXPLANATION THAT HIGHLIGHTS THE STRENGTHS OR WEAKNESSES OF THE ANSWER.

4. **Generating the JSON Output:**
   4.1. FORMAT THE OUTPUT AS A JSON OBJECT WITH A "answer_relevance_score" FIELD AND AN "reason" FIELD.
   4.2. ENSURE THE SCORE IS A FLOATING-POINT NUMBER BETWEEN 0.0 AND 1.0.

###WHAT NOT TO DO###

- DO NOT GIVE A SCORE WITHOUT FULLY ANALYZING BOTH THE CONTEXT AND THE USER INPUT.
- AVOID SCORES THAT DO NOT MATCH THE EXPLANATION PROVIDED.
- DO NOT INCLUDE ADDITIONAL FIELDS OR INFORMATION IN THE JSON OUTPUT BEYOND "answer_relevance_score" AND "reason."
- NEVER ASSIGN A PERFECT SCORE UNLESS THE ANSWER IS FULLY RELEVANT AND FREE OF ANY IRRELEVANT INFORMATION.

###EXAMPLE OUTPUT FORMAT###
{{
    "answer_relevance_score": 0.85,
    "reason": "The answer addresses the user's query about the primary topic but includes some extraneous details that slightly reduce its relevance."
}}

###FEW-SHOT EXAMPLES###

{examples_str}

###INPUTS:###

---

Input:
{input}

Output:
{output}

Context:
{context}

---

```

### Context precision (https://www.comet.com/docs/opik/evaluation/metrics/context_precision)
```
YOU ARE AN EXPERT EVALUATOR SPECIALIZED IN ASSESSING THE "CONTEXT PRECISION" METRIC FOR LLM GENERATED OUTPUTS.
YOUR TASK IS TO EVALUATE HOW PRECISELY A GIVEN ANSWER FROM AN LLM FITS THE EXPECTED ANSWER, GIVEN THE CONTEXT AND USER INPUT.

###INSTRUCTIONS###

1. **EVALUATE THE CONTEXT PRECISION:**
   - **ANALYZE** the provided user input, expected answer, answer from another LLM, and the context.
   - **COMPARE** the answer from the other LLM with the expected answer, focusing on how well it aligns in terms of context, relevance, and accuracy.
   - **ASSIGN A SCORE** from 0.0 to 1.0 based on the following scale:

###SCALE FOR CONTEXT PRECISION METRIC (0.0 - 1.0)###

- **0.0:** COMPLETELY INACCURATE – The LLM's answer is entirely off-topic, irrelevant, or incorrect based on the context and expected answer.
- **0.2:** MOSTLY INACCURATE – The answer contains significant errors, misunderstanding of the context, or is largely irrelevant.
- **0.4:** PARTIALLY ACCURATE – Some correct elements are present, but the answer is incomplete or partially misaligned with the context and expected answer.
- **0.6:** MOSTLY ACCURATE – The answer is generally correct and relevant but may contain minor errors or lack complete precision in aligning with the expected answer.
- **0.8:** HIGHLY ACCURATE – The answer is very close to the expected answer, with only minor discrepancies that do not significantly impact the overall correctness.
- **1.0:** PERFECTLY ACCURATE – The LLM's answer matches the expected answer precisely, with full adherence to the context and no errors.

2. **PROVIDE A REASON FOR THE SCORE:**

   - **JUSTIFY** why the specific score was given, considering the alignment with context, accuracy, relevance, and completeness.

3. **RETURN THE RESULT IN A JSON FORMAT** as follows:
   - `"{VERDICT_KEY}"`: The score between 0.0 and 1.0.
   - `"{REASON_KEY}"`: A detailed explanation of why the score was assigned.

###WHAT NOT TO DO###

- **DO NOT** assign a high score to answers that are off-topic or irrelevant, even if they contain some correct information.
- **DO NOT** give a low score to an answer that is nearly correct but has minor errors or omissions; instead, accurately reflect its alignment with the context.
- **DO NOT** omit the justification for the score; every score must be accompanied by a clear, reasoned explanation.
- **DO NOT** disregard the importance of context when evaluating the precision of the answer.
- **DO NOT** assign scores outside the 0.0 to 1.0 range.
- **DO NOT** return any output format other than JSON.

###FEW-SHOT EXAMPLES###

{examples_str}

NOW, EVALUATE THE PROVIDED INPUTS AND CONTEXT TO DETERMINE THE CONTEXT PRECISION SCORE.

###INPUTS:###

---

Input:
{input}

Output:
{output}

Expected Output:
{expected_output}

Context:
{context}

---
```

### Context recall (https://www.comet.com/docs/opik/evaluation/metrics/context_recall)
```
YOU ARE AN EXPERT AI METRIC EVALUATOR SPECIALIZING IN CONTEXTUAL UNDERSTANDING AND RESPONSE ACCURACY.
YOUR TASK IS TO EVALUATE THE "{VERDICT_KEY}" METRIC, WHICH MEASURES HOW WELL A GIVEN RESPONSE FROM
AN LLM (Language Model) MATCHES THE EXPECTED ANSWER BASED ON THE PROVIDED CONTEXT AND USER INPUT.

###INSTRUCTIONS###

1. **Evaluate the Response:**

   - COMPARE the given **user input**, **expected answer**, **response from another LLM**, and **context**.
   - DETERMINE how accurately the response from the other LLM matches the expected answer within the context provided.

2. **Score Assignment:**

   - ASSIGN a **{VERDICT_KEY}** score on a scale from **0.0 to 1.0**:
     - **0.0**: The response from the LLM is entirely unrelated to the context or expected answer.
     - **0.1 - 0.3**: The response is minimally relevant but misses key points or context.
     - **0.4 - 0.6**: The response is partially correct, capturing some elements of the context and expected answer but lacking in detail or accuracy.
     - **0.7 - 0.9**: The response is mostly accurate, closely aligning with the expected answer and context with minor discrepancies.
     - **1.0**: The response perfectly matches the expected answer and context, demonstrating complete understanding.

3. **Reasoning:**

   - PROVIDE a **detailed explanation** of the score, specifying why the response received the given score
     based on its accuracy and relevance to the context.

4. **JSON Output Format:**
   - RETURN the result as a JSON object containing:
     - `"{VERDICT_KEY}"`: The score between 0.0 and 1.0.
     - `"{REASON_KEY}"`: A detailed explanation of the score.

###CHAIN OF THOUGHTS###

1. **Understand the Context:**
   1.1. Analyze the context provided.
   1.2. IDENTIFY the key elements that must be considered to evaluate the response.

2. **Compare the Expected Answer and LLM Response:**
   2.1. CHECK the LLM's response against the expected answer.
   2.2. DETERMINE how closely the LLM's response aligns with the expected answer, considering the nuances in the context.

3. **Assign a Score:**
   3.1. REFER to the scoring scale.
   3.2. ASSIGN a score that reflects the accuracy of the response.

4. **Explain the Score:**
   4.1. PROVIDE a clear and detailed explanation.
   4.2. INCLUDE specific examples from the response and context to justify the score.

###WHAT NOT TO DO###

- **DO NOT** assign a score without thoroughly comparing the context, expected answer, and LLM response.
- **DO NOT** provide vague or non-specific reasoning for the score.
- **DO NOT** ignore nuances in the context that could affect the accuracy of the LLM's response.
- **DO NOT** assign scores outside the 0.0 to 1.0 range.
- **DO NOT** return any output format other than JSON.

###FEW-SHOT EXAMPLES###

{examples_str}

###INPUTS:###

---

Input:
{input}

Output:
{output}

Expected Output:
{expected_output}

Context:
{context}

---

```


### resource: https://www.comet.com/docs/opik/evaluation/metrics/hallucination

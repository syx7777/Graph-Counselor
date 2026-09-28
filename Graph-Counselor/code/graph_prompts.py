from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

GRAPH_DEFINITION = {'maple': 'There are three types of nodes in the graph: paper, author and venue.\nPaper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.\nPaper nodes are linked to author nodes, venue nodes, reference nodes and cited_by nodes. Author nodes are linked to paper nodes. Venue nodes are linked to paper nodes.',
                    'biomedical': 'There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.\nEach node has name feature.\nThere are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.',
                    'legal': 'There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.\nOpinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.\nOpinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.',
                    'amazon': 'There are two types of nodes in the graph: item and brand.\nItem nodes have features: title, description, price, img, category. Brand nodes have features: name.\nItem nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.',
                    'goodreads': 'There are four types of nodes in the graph: book, author, publisher, and series.\nBook nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description.\nBook nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.',
                    'dblp': 'There are three types of nodes in the graph: paper, author and venue.\nPaper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.\nPaper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.'}

# HEADER
REFLECTION_HEADER = 'You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n'
REFLECTION_AFTER_LAST_TRIAL_HEADER = 'You have attempted to answer the following question before and failed. The following reflection(s) help you avoid repeating the same mistakes made in your previous attempt. Use them to improve your strategy of correctly answering the given question.\n'
LAST_TRIAL_HEADER = 'You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question. Additionally, in your next round of reasoning, you can directly use the information retrieved from the graph here to reduce the reasoning steps. \n'

GraphAgent_INSTRUCTION = """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Here are some examples:
{examples}
(END OF EXAMPLES)
When last Observation has been given or there is no Thought, you should provide next only one Thought based on the question. When last Thought has been given, you should provide next only one Action. When you think it's time to finish, use Finish to end the process. Don't make Observation. The new text you generate should be in a line and less than 512 tokens.
Definition of the graph: {graph_definition}
Question: {question} Please answer by providing node main feature (e.g., names) rather than node IDs. {scratchpad}"""

GraphAgent_INSTRUCTION_COMPOUND = """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Besides, you can use compound function, such as Feature[Retrieve[keyword], feature], which returns the detailed attribute information of Retrieve[keyword] regarding the given "feature" key.
Here are some examples:
{examples}
(END OF EXAMPLES)
When last Observation has been given or there is no Thought, you should provide next only one Thought based on the question. When last Thought has been given, you should provide next only one Action. When you think it's time to finish, use Finish to end the process. Don't make Observation.
Definition of the graph: {graph_definition}
Question: {question} Please answer by providing node main feature (e.g., names) rather than node IDs. {scratchpad}"""

REFLECT_GraphAgent_INSTRUCTION_COMPOUND = """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Besides, you can use compound function, such as Feature[Retrieve[keyword], feature], which returns the detailed attribute information of Retrieve[keyword] regarding the given "feature" key.
Here are some examples:
{examples}
(END OF EXAMPLES)
{reflections}
When last Observation has been given or there is no Thought, you should provide next only one Thought based on the question. When last Thought has been given, you should provide next only one Action. When you think it's time to finish, use Finish to end the process. Don't make Observation.
Definition of the graph: {graph_definition}
Question: {question} Please answer by providing node main feature (e.g., names) rather than node IDs. {scratchpad}"""


GraphAgent_INSTRUCTION_ZeroShot = """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Definition of the graph: {graph_definition}
Question: {question} Please answer by providing node main feature (e.g., names) rather than node IDs. {scratchpad}"""

REFLECT_GraphAgent_INSTRUCTION = """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Here are some examples:
{examples}
(END OF EXAMPLES)
{reflections}
Definition of the graph: {graph_definition}
Question: {question} Please answer by providing node main feature (e.g., names) rather than node IDs. {scratchpad}"""

REFLECT_GraphAgent_INSTRUCTION_ZeroShot = """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
{reflections}
Definition of the graph: {graph_definition}
Question: {question} Please answer by providing node main feature (e.g., names) rather than node IDs. {scratchpad}"""


REFLECT_Agent_INSTRUCTION_BASE = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to specified graph function tools and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[], you used up your set number of reasoning steps, or the total length of your reasoning is over the limit.  In a few sentences, diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences. 
Here are some examples:
{examples}
(END OF EXAMPLES)
Previous trial:
Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and in Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Definition of the graph: {graph_definition}
Question: {question}Please answer by providing node main feature (e.g., names) rather than node IDs.{scratchpad}
Reflection:"""

REFLECT_Agent_INSTRUCTION_SHORT_MULTIPLE = """Please write the Reflections, including the content for Recap of the Trial, Guided Reflection, based on the guidance provided in Graph Function Background, Previous Trial Details, Recap of the Trial, and Guided Reflection.
Graph Function Background
- Definition of the graph: {graph_definition}
- You were provided with the following functions to interact with the graph:
  - Retrieve(keyword): Finds the related node based on the query keyword.
  - Feature(Node, feature): Retrieves detailed attribute information for the specified node and feature key.
  - Degree(Node, neighbor_type): Calculates the number of neighbors of the specified type for the given node.
  - Neighbor(Node, neighbor_type): Lists the neighbors of the specified type for the given node.
Recap of the Trial
- Question: [Insert the problem description here]
- Graph Information Used: [List the graph structural information selected in the trial]
- Outcome: The question remained unanswered due to:
  - Incorrect assumptions or guessing.
  - Exhausted reasoning steps or exceeded the reasoning limit.
Guided Reflection
- Understanding the Question
  - Core Goal: What is the main objective of this question?
  - Missed Details: Were any critical keywords or relationships overlooked?
  - Misinterpretations: Identify any misunderstandings and correct them.
-  Analysis of Selected Graph Information
  - Relevance: Why did you choose the graph data used? Was it aligned with the goal?
  - Missed Insights: Were there other relevant pieces of information you didn't consider? If so, why?
  - Redundancies: Flag any irrelevant information that added confusion or wasted reasoning steps.
- Align the problem understanding and graph function information
  - Assess whether your understanding of the problem matched the graph selection. Identify inconsistencies and refine your selection criteria to better suit the task.
- Improved Strategy
Based on your reflection:
  - Updated Understanding of the Problem: Revise and describe your updated understanding.
  - Revised Graph Selection: List and explain which graph information you would now choose and why it is more suitable.
  - Avoiding Past Issues: Describe how this strategy addresses the challenges and improves your reasoning.
Here are some examples:
{examples}
(END OF EXAMPLES)
Previous trial:
Question: {question}
{scratchpad}
Reflection:
"""


REFLECT_Agent_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self reflection. Reflect on your prior reasoning trial, and find areas for improvement to enhance your performance in answering the question next time. Please write the Reflections, including the content for Recap of the Trial, Guided Reflection, based on the guidance provided in Graph Function Background, Previous Trial Details, Recap of the Trial, and Guided Reflection.
Graph Function Background
- Definition of the graph: {graph_definition}
- You were provided with the following functions to interact with the graph:
  - Retrieve(keyword): Finds the related node based on the query keyword.
  - Feature(Node, feature): Retrieves detailed attribute information for the specified node and feature key.
  - Degree(Node, neighbor_type): Calculates the number of neighbors of the specified type for the given node.
  - Neighbor(Node, neighbor_type): Lists the neighbors of the specified type for the given node.
Recap of the Trial
- Question: [Insert the problem description here]
- Graph Information Used: [List the graph structural information selected in the trial]
- Outcome: The question was not successfully answered due to one of the following reasons:
  - You guessed the wrong answer with Finish[].
  - You used up the set number of reasoning steps (10 steps).
  - Your total reasoning length exceeded the limit.
Guided Reflection
- Understanding the Question
  - Core Goal: What is the main objective of this question?
  - Missed Information: Could you have overlooked any critical details?
  - Potential Misunderstandings: Were there any misinterpretations in your approach? If so, list and correct them.
-  Analysis of Selected Graph Information
  - Relevance: Why did you choose the information you selected? How did it help answer the question?
  - Missed Insights: Were there other relevant pieces of information you didn't consider? If so, why?
  - Redundancies: Did you include irrelevant or redundant information? If yes, identify and revise.
- Align the problem understanding and graph function information
  - Understanding and Selection: How did your understanding of the problem influence your graph structure choice?
  - Inconsistencies: Were there any mismatches between your understanding and your graph structure selection? If so, what caused them?
  - Adjustments: How can you better align your understanding with the graph structure selection?
- Improved Strategy
Based on your reflection:
  - Updated Understanding of the Problem: Revise and describe your updated understanding.
  - Revised Graph Selection: List and explain which graph information you would now choose and why it is more suitable.
  - Avoiding Past Issues: Describe how this strategy addresses the challenges and improves your reasoning.
Here are some examples:
{examples}
(END OF EXAMPLES)
Previous trial Details:
Question: {question} Please answer by providing node main feature (e.g., names) rather than node IDs.
{scratchpad}
Reflection:
"""

COUMPOUND_REFLECT_Agent_INSTRUCTION= """You are an advanced reasoning agent that can improve based on self reflection. Reflect on your prior reasoning trial, and find areas for improvement to enhance your performance in answering the question next time. Please write the Reflections, including the content for Recap of the Trial, Guided Reflection, based on the guidance provided in Graph Function Background, Previous Trial Details, Recap of the Trial, and Guided Reflection.
Graph Function Background
- Definition of the graph: {graph_definition}
- You were provided with the following functions to interact with the graph:
  - Retrieve(keyword): Finds the related node based on the query keyword.
  - Feature(Node, feature): Retrieves detailed attribute information for the specified node and feature key.
  - Degree(Node, neighbor_type): Calculates the number of neighbors of the specified type for the given node.
  - Neighbor(Node, neighbor_type): Lists the neighbors of the specified type for the given node.
  Besides, you can use compound function, such as Feature[Retrieve[keyword], feature], which returns the detailed attribute information of Retrieve[keyword] regarding the given "feature" key.
Recap of the Trial
- Question: [Insert the problem description here]
- Graph Information Used: [List the graph structural information selected in the trial]
- Outcome: The question was not successfully answered due to one of the following reasons:
  - You guessed the wrong answer with Finish[].
  - You used up the set number of reasoning steps (10 steps).
  - Your total reasoning length exceeded the limit.
Guided Reflection
- Understanding the Question
  - Core Goal: What is the main objective of this question?
  - Missed Information: Could you have overlooked any critical details?
  - Potential Misunderstandings: Were there any misinterpretations in your approach? If so, list and correct them.
-  Analysis of Selected Graph Information
  - Relevance: Why did you choose the information you selected? How did it help answer the question?
  - Missed Insights: Were there other relevant pieces of information you didn't consider? If so, why?
  - Redundancies: Did you include irrelevant or redundant information? If yes, identify and revise.
- Align the problem understanding and graph function information
  - Understanding and Selection: How did your understanding of the problem influence your graph structure choice?
  - Inconsistencies: Were there any mismatches between your understanding and your graph structure selection? If so, what caused them?
  - Adjustments: How can you better align your understanding with the graph structure selection?
- Improved Strategy
Based on your reflection:
  - Updated Understanding of the Problem: Revise and describe your updated understanding.
  - Revised Graph Selection: List and explain which graph information you would now choose and why it is more suitable. Explain how these graph functions can be combined into composite functions to streamline operations, ensuring no more than two functions are combined at each step.
  - Avoiding Past Issues: Describe how this strategy addresses the challenges and improves your reasoning.
Here are some examples:
{examples}
(END OF EXAMPLES)
Previous trial Details:
Question: {question} Please answer by providing node main feature (e.g., names) rather than node IDs.
{scratchpad}
Reflection:
"""

COUMPOUND_PLAN_REFLECT_Agent_INSTRUCTION_BASE=""""""
COUMPOUND_PLAN_REFLECT_Agent_INSTRUCTION_SHORT_MULTIPLE=""""""
COUMPOUND_REFLECT_Agent_INSTRUCTION_BASE="""

"""
COUMPOUND_REFLECT_Agent_INSTRUCTION_SHORT_MULTIPLE=""""""

PLAN_GraphAgent_INSTRUCTION = """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Plan step, you can think about what the question is asking and plan how to do to get the answer. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Besides, you can use compound function, such as Feature[Retrieve[keyword], feature], which returns the detailed attribute information of Retrieve[keyword] regarding the given "feature" key.
Here are some examples:
{examples}
(END OF EXAMPLES)
When last Observation has been given or there is no Plan, you should provide next only one Plan based on the question. When last Plan has been given, you should provide next only one Thought. When last Thought has been given, you should provide next only one Action. When you think it's time to finish, use Finish to end the process. Don't make Observation.
Definition of the graph: {graph_definition}
Question: {question} Please answer by providing node main feature (e.g., names) rather than node IDs. {scratchpad}"""

REFLECT_PLAN_GraphAgent_INSTRUCTION = """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Plan step, you can think about what the question is asking and plan how to do to get the answer. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Here are some examples:
{examples}
(END OF EXAMPLES)
{reflections}
When last Observation has been given or there is no Plan, you should provide next only one Plan based on the question. When last Plan has been given, you should provide next only one Thought. When last Thought has been given, you should provide next only one Action. When you think it's time to finish, use Finish to end the process. Don't make Observation.
Definition of the graph: {graph_definition}
Question: {question} Please answer by providing node main feature (e.g., names) rather than node IDs. {scratchpad}"""

REFLECT_PLAN_GraphAgent_NEW_INSTRUCTION = """Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Plan step, you can think about what the question is asking and plan how to do to get the answer. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Besides, you can use compound function, such as Feature[Retrieve[keyword], feature], which returns the detailed attribute information of Retrieve[keyword] regarding the given "feature" key.
You can't use any functions other than the four mentioned above. You can't use a combination of more than two functions.
For straightforward questions, prioritize simple actions and avoid unnecessary complexity.
Here are some examples:
{examples}
(END OF EXAMPLES)
{reflections}
When last Observation has been given or there is no Plan, you should provide next only one Plan based on the question. When last Plan has been given, you should provide next only one Thought. When last Thought has been given, you should provide next only one Action. When you think it's time to finish, use Finish to end the process. Don't make Observation.
Definition of the graph: {graph_definition}
Question: {question} Please answer by providing node main feature (e.g., names) rather than node IDs. {scratchpad}"""



EVAL_Agent_INSTRUCTION = """You are an intelligent reasoning accuracy evaluation agent. Evaluate the final answer based on all the thought, action, and observation processes and determine if it meets the problem requirements. Ensure the following: The final answer directly corresponds to the data retrieved from the graph. It satisfies the question's requirement without including irrelevant or incorrect elements. The reasoning behind the answer is logical and supported by the observations. In a few sentences, please provide a brief explanation summarizing why the answer meets or does not meet the criteria. Then, please conclude with a clear judgment based on the  explanation, respond [yes] if the answer is correct , or [no] if the answer is not correct.
Here are some examples:
{examples}
(END OF EXAMPLES)
Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Definition of the graph: {graph_definition}
Question: {question}Please answer by providing node main feature (e.g., names) rather than node IDs.{scratchpad}

Proceed with explanation and judgment below:
"""

EVAL_COMPOUND_PLAN_Agent_INSTRUCTION = """You are an intelligent reasoning accuracy evaluation agent. Evaluate the final answer based on all the plan, thought, action, and observation processes and determine if it meets the problem requirements. Ensure the following: The final answer directly corresponds to the data retrieved from the graph. It satisfies the question's requirement without including irrelevant or incorrect elements. The reasoning behind the answer is logical and supported by the observations. In a few sentences, please provide a brief explanation summarizing why the answer meets or does not meet the criteria. Then, please conclude with a clear judgment based on the  explanation, respond [yes] if the answer is correct , or [no] if the answer is not correct.
Here are some examples:
{examples}
(END OF EXAMPLES)
Solve a question answering task with interleaving Thought, Interaction with Graph, Feedback from Graph steps. In Plan step, you can think about what the question is asking and plan how to do to get the answer. In Thought step, you can think about what further information is needed, and In Interaction step, you can get feedback from graphs with four functions: 
(1) Retrieve[keyword], which retrieves the related node from the graph according to the corresponding query.
(2) Feature[Node, feature], which returns the detailed attribute information of Node regarding the given "feature" key.
(3) Degree[Node, neighbor_type], which calculates the number of "neighbor_type" neighbors of the node Node in the graph.
(4) Neighbor[Node, neighbor_type], which lists the "neighbor_type" neighbours of the node Node in the graph and returns them.
Besides, you can use compound function, such as Feature[Retrieve[keyword], feature], which returns the detailed attribute information of Retrieve[keyword] regarding the given "feature" key.
When last Observation has been given or there is no Plan, you should provide next only one Plan based on the question. When last Plan has been given, you should provide next only one Thought. When last Thought has been given, you should provide next only one Action. When you think it's time to finish, use Finish to end the process. Don't make Observation.
Definition of the graph: {graph_definition}
Question: {question}Please answer by providing node main feature (e.g., names) rather than node IDs.{scratchpad}

Proceed with explanation and judgment below:
"""
EVAL_COMPOUND_Agent_INSTRUCTION = """
"""


graph_agent_prompt_zeroshot = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", GraphAgent_INSTRUCTION_ZeroShot),
])

graph_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. Answer step by step."),
    ("human", GraphAgent_INSTRUCTION),
])

reflect_graph_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", REFLECT_GraphAgent_INSTRUCTION),
])

reflect_graph_agent_prompt_zeroshot = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. "),
    ("human", REFLECT_GraphAgent_INSTRUCTION_ZeroShot),
])

graph_reflect_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", REFLECT_Agent_INSTRUCTION),
])

graph_reflect_prompt_base = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", REFLECT_Agent_INSTRUCTION_BASE),
])

graph_reflect_prompt_short_multiple = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", REFLECT_Agent_INSTRUCTION_SHORT_MULTIPLE),
])

graph_eval_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", EVAL_Agent_INSTRUCTION),
])

graph_compound_and_plan_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", PLAN_GraphAgent_INSTRUCTION),
])

graph_compound_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", GraphAgent_INSTRUCTION_COMPOUND),
])

reflect_graph_compound_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", REFLECT_GraphAgent_INSTRUCTION_COMPOUND),
])

reflect_graph_compound_and_plan_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", REFLECT_PLAN_GraphAgent_INSTRUCTION),
])

reflect_graph_compound_and_new_plan_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", REFLECT_PLAN_GraphAgent_NEW_INSTRUCTION),
])

graph_compound_and_plan_reflect_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", COUMPOUND_REFLECT_Agent_INSTRUCTION),
])

graph_compound_and_plan_reflect_prompt_base = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", COUMPOUND_PLAN_REFLECT_Agent_INSTRUCTION_BASE),
])

graph_compound_and_plan_reflect_prompt_short_multiple = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", COUMPOUND_PLAN_REFLECT_Agent_INSTRUCTION_SHORT_MULTIPLE),
])

graph_compound_reflect_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", COUMPOUND_REFLECT_Agent_INSTRUCTION),
])


graph_compound_reflect_prompt_base = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", COUMPOUND_REFLECT_Agent_INSTRUCTION_BASE),
])

graph_compound_reflect_prompt_short_multiple = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", COUMPOUND_REFLECT_Agent_INSTRUCTION_SHORT_MULTIPLE),
])

graph_compound_and_plan_eval_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", EVAL_COMPOUND_PLAN_Agent_INSTRUCTION),
])

graph_compound_eval_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", EVAL_COMPOUND_Agent_INSTRUCTION),
])
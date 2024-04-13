# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:53:22 2024

@author: Nova18
"""

import pandas as pd

def data_preprocessing(file):
    df = file['train'][::]
    questions, answers, image_ids, images = [], [], [], []
    # Iterate through the conversations and flatten them into questions and answers
    for index, conversation in enumerate(df['conversations']):
        for i in range(0, len(conversation), 2):         # Assuming there is always an answer for each question
            questions.append(conversation[i]['value'])
            answers.append(conversation[i+1]['value'])
            image_ids.append(df['id'][index])
            images.append(df['image'][index])
    
    dataset = pd.DataFrame({
                            'question': questions,
                            'answer': answers,
                            'image_id': image_ids,
                            'image': images
                            })
    return dataset


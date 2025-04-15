from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from vllm import LLM, SamplingParams
from tqdm import tqdm
import asyncio
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
modelpath = "/home/liuzhou/projects/Rare_rag/Qwen2.5-7B-Instruct"

def load_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(
        modelpath,
        sliding_window=None,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=False  # 显存优化
    )

    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    print("✅ 模型加载成功")
    return model, tokenizer


def generate_responses_transformers(prompts, model, tokenizer, max_new_tokens=2048):
    # 编码输入
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    print("模型输入:", inputs)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,      #
        num_beams=1,          #
        temperature=None,     #
        top_p=None,           #
        top_k=None            #
    )

    print("\n✅ 模型原始输出张量:", outputs)

    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return responses


# 示例用法
if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    input_text = """ various tissue dysfunctions \\[[@B128]\\]. At present, drugs for the treatment of this disease are urgently needed to be developed. As confirmed by the experiment, extracts from licorice have a protective effect on diabetic nephropathy \\[[@B129]\\].\n\nThrough the oral glucose tolerance test, the hypoglycemic effect of isoliquiritigenin on normal Swiss albino male mice was reported \\[[@B107]\\]. Another finding demonstrated that isoliquiritigenin diminished high glucose-induced mesangial matrix accumulation through retarding transforming growth factor (TGF)-*β*1-SMAD signaling transduction \\[[@B108]\\]. Angiotensin-converting enzyme (ACE) plays a prominent role in hypertension, heart failures, myocardial infarction, and diabetic nephropathy. The study showed that echinatin has been proved to show a certain inhibitory effect on ACE *in vitro* \\[[@B109]\\]. Furthermore, licochalcone E enhanced expression of PPAR-*γ* through irritating Akt signals as well as functions as a PPAR-*γ* partial agonist, which improved hyperlipidemia and hyperglycemia under diabetic conditions \\[[@B12]\\]. Kanzonol C, licoagrochalcone A, and isobavachalcone as inhibitors of protein tyrosine phosphate 1B (PTP1B) were potential candidates for treating type II diabetes \\[[@B21], [@B110]\\].\n\n3.8. Antiobesity Activity {#sec3.8}\n-------------------------\n\nObesity is a globally epidemic chronic metabolic disease, and the proportion of obese people continues to rise due to changes in lifestyle and diet. Obesity poses a series of potential safety hazard, so the use of antiobesity drugs can help improve the health of patients. Kanzonol C, licoagrochalcone A, and isobavachalcone were found to be PTP1B inhibitors for treatment of obesity \\[[@B21], [@B110]\\]. Isoliquiritin apioside and isoliquiritigenin, as sources of pancreatic lipase (PL) inhibitors for preventing obesity, could lower the plasma total triglycerides and total cholesterol \\[[@B111]\\]. Licochalcone A had an inhibitory effect on adipocyte differentiation and lipogenesis via the downregulation of PPAR-*γ*, CCAAT/enhancer binding protein *α* (C/EBP*α*), and sterol regulatory element-binding protein 1c (SREBP-1c) in 3T3-L1 preadipocytes \\[[@B112]\\]. And other results demonstrated that licochalcone A was effective to reduce obesity and could recover metabolic homeostasis by inducing adipocyte browning \\[[@B113]\\].\n\n3.9. Other Activities {#sec3.9}\n---------------------\n\nIsoliquiritigenin has been detected to have antiplatelet action \\[[@B130]\\], protective effect on cerebral ischemia injury \\[[@B131]\\], and estrogen-like \\[[@B132]\\], neuroprotective \\[[@B133]\\], and antimelanogenic \\[[@B134]\\] activities. Licochalcone A has been demonstrated to possess antispasmodic \\[[@B135]\\], antileishmanial \\[[@B136]\\], antimalarial \\[[@B137]\\], and osteogenic activities \\[[@B138]\\]. Isoliquiritin was studied to produce significant antidepressant-like effect \\[[@B13]\\].\n\n4. Conclusion {#sec4}\n=============\n\nPhytochemical constituents especially flavonoids are largely considered to be beneficial for human health and disease prevention. As a category of nontoxic and effective natural ingredients, chalcones are proved to possess lots of biological activities and medicinal properties. To date, about 42 chalcones in licorice have been isolated and identified, and more new structures will be unveiled. Meanwhile, most of chalcones in licorice have been widely and deeply studied for their various activities, such as anticancer, anti-inflammatory, antimicrobial, antiviral, antioxidative, hepatoprotective, antidiabetic, and antiobesity activities. However, it will be a long way to further validate the pharmacological action and develop new drug. As chalcones in licorice are deeply explored and fully utilized, it will be served as a broad prospect for development and utilization of licorice.\n\nThis work was supported by the National Natural Science Foundation of China (81873192), the Science & Technology Development Fund of Tianjin Education Commission for Higher Education (2018ZD02), and National Key Research and Development Project of China (2018YFC1707905).\n\nConflicts of Interest\n=====================\n\nThe authors declare that there are no conflicts of interest regarding the publication of this paper.\n\nAuthors\\' Contributions\n=======================\n\nDanni Wang and Jing Liang contributed equally to this work.\n\n![Basic framework of chalcone and dihydrochalcone.](ECAM2020-3821248.001){#fig1}\n\n![The structures of chalcones from licorice.](ECAM2020-3821248.002){#fig2}\n\n![The biological activities of chalcones from licorice.](ECAM2020-3821248.003){#fig3}\n\n###### \n\nThe sources of chalcones from licorice.\n\n  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------\n  Number   Name                                                                                     Source                   Reference\n  -------- ---------------------------------------------------------------------------------------- ------------------------ --------------------------------------------\n  C1       Isoliquiritin apioside                                                                   *G. glabra* L.\\          \\[[@B16], [@B17]\\]\n                                                                                                    *G. uralensis* Fisch.\\   \n                                                                                                    *G. inflata* Bat.        \n\n  C2       Licuraside                                                                               *G. glabra* L.\\          \\[[@B17]\\]\n                                                                                                    *G. uralensis* Fisch.\\   \n                                                                                                    *G. inflata* Bat.        \n\n  C3       Isoliquiritin                                                                            *G. glabra* L.\\          \\[[@B16]--[@B18]\\]\n                                                                                                    *G. uralensis* Fisch.\\   \n                                                                                                    *G. inflata* Bat.        \n\n  C4       Butein-4-*O*-*β*-D-glucopyranoside                                                       *G. uralensis* Fisch.    \\[[@B19]\\]\n\n  C5       Neoisoliquiritin                                                                         *G. glabra* L.\\          \\[[@B14], [@B16], [@B20]\\]\n                                                                                                    *G. uralensis* Fisch.\\   \n                                                                                                    *G. inflata* Bat.        \n\n  C6       Isoliquiritigenin                                                                        *G. glabra* L.\\          \\[[@B16], [@B17], [@B21]--[@B24]\\]\n                                                                                                    *G. uralensis* Fisch.\\   \n                                                                                                    *G. inflata* Bat.        \n\n  C7       Homobutein                                                                               *G. uralensis* Fisch.    \\[[@B18]\\]\n\n  C8       Echinatin                                                                                *G. uralensis* Fisch.\\   \\[[@B10], [@B18], [@B21], [@B22], [@B25]\\]\n                                                                                                    *G. glabra* L.\\          \n                                                                                                    *G. inflata* Bat.        \n\n  C9       Licochalcone A                                                                           *G. glabra* L.\\          \\[[@B10], [@B14], [@B17]\\]\n                                                                                                    *G. uralensis* Fisch.\\   \n                                                                                                    *G. inflata* Bat.        \n\n  C10      Licochalcone B                                                                           *G. uralensis* Fisch.\\"""
    prompts = [
        f"""请清理以下文本，移除所有引用标记（如[@b12]）、URL、Latex标签（如\\usepackage）,无关符号 等无关内容：
        ------------
        {input_text}
        ------------
        直接返回清理后的结果，不要包含额外解释。"""
    ]
    responses = generate_responses_transformers(prompts, model, tokenizer)
    for prompt, response in zip(prompts, responses):
        print(f"Prompt: {prompt}\nResponse: {response}\n{'-' * 50}")


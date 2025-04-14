import os
from openai import OpenAI
import openai
import time
import traceback
from datetime import datetime


class GPT3Assistant:
    def __init__(self):

        self.client = OpenAI(api_key="xxx", base_url="xxx")

    def ask_question(self, question):
        answer = None
        while answer is None:
            try:
                completion = self.client.chat.completions.create(
                    model="your deepseek models",
                    messages=[
                        {"role": "system",
                         "content": "You are a programming expert proficient in various languages and algorithms. Please do not output any interpretation other than the code."},
                        {"role": "user", "content": question}
                    ]
                )
                answer = completion.choices[0].message.content
            except openai.RateLimitError as e:
                print("Rate limit exceeded. Waiting...")
                time.sleep(10)  # wait for a minute
            except openai.APIConnectionError as e:
                print("API connection error. Waiting...")
                answer = 'error1'
                # time.sleep(5)  # wait for a minute
            except Exception as e:
                traceback.print_exc()
                print("Unknown error. Waiting...")
                answer = 'error2'
                # time.sleep(5)  # wait for a minute
        return answer


def abstract_the_code(code, ask):
    ques = ask + code
    assistant = GPT3Assistant()
    a2 = assistant.ask_question(ques)
    return a2


prompt_delete = """
Please remove the death code in the following code:
"""

prompt_java = """ 
Replace all variables in the code of the form "var + number" or "VAR + number" such as var1 ，the new variables should use more specific identifiers according to the semantics of the code.
In addition, you also need to replace the function name "methodName"
At implementation time, you can check token by token and perform substitutions, but you should make sure that the same token in different locations should be replaced with the same new token.
here is an example：
# code before
```java
public static String methodName(ClassLoader VAR0, String VAR1, boolean VAR2) throws IOException {
        if (VAR2) {
            StringMaker VAR3 = new StringMaker();
            Enumeration VAR4 = VAR0.getResources(VAR1);
            while (VAR4.hasMoreElements()) {
                URL VAR5 = (URL) VAR4.nextElement();
                InputStream VAR6 = VAR5.openStream();
                String VAR7 = reader(VAR6);
                if (VAR7 != null) {
                    VAR3.append(VAR7);
                    VAR3.append(StringPool.NEW_LINE);
                }
                VAR6.close();
            }
            return VAR3.toString().trim();
        } else {
            InputStream VAR8 = VAR0.getResourceAsStream(VAR1);
            String VAR9 = reader(VAR8);
            VAR8.close();
            return VAR9;
        }
    }
```
# code after
```java
public static String readResources(ClassLoader classLoader, String resourceName, boolean isMultiple) throws IOException {
    if (isMultiple) {
        StringMaker stringMaker = new StringMaker();
        Enumeration resources = classLoader.getResources(resourceName);
        while (resources.hasMoreElements()) {
            URL url = (URL) resources.nextElement();
            InputStream inputStream = url.openStream();
            String content = reader(inputStream);
            if (content != null) {
                stringMaker.append(content);
                stringMaker.append(StringPool.NEW_LINE);
            }
            inputStream.close();
        }
        return stringMaker.toString().trim();
    } else {
        InputStream inputStream = classLoader.getResourceAsStream(resourceName);
        String content = reader(inputStream);
        inputStream.close();
        return content;
    }
}
```
According to the example above，Please process the flowing code:
"""

prompt_c = """ 
Replace all variables in the code of the form "var + number" or "VAR + number" such as var1 ，the new variables should use more specific identifiers according to the semantics of the code.
In addition, you also need to replace the function name "methodName"
At implementation time, you can check token by token and perform substitutions, but you should make sure that the same token in different locations should be replaced with the same new token.
here is an example：
# code before
```c
int main(void)
{
  int var2 = 1;
  int var1 = 0;
  scanf("%d", &var2);
  while (var2 > 0)
  {
    int var3;
    int var4;
    int var5;
    scanf("%d", &var3);
    int var6[var3][var3];
    for (var4 = 0; var4 < var3; var4++)
    {
      for (var5 = 0; var5 < var3; var5++)
      {
        var6[var4][var5] = 0;
      }
    }
    for (var4 = 0; var4 < var3; var4++)
    {
      for (var5 = 0; var5 <= var4; var5++)
      {
        scanf("%d", &var6[var4][var5]);
      }
    }
    for (var4 = 1; var4 < var3; var4++)
    {
      for (var5 = 0; var5 <= var4; var5++)
      {
        if (var5 == 0)
        {
          var6[var4][var5] = var6[var4][var5] + var6[var4 - 1][var5];
        }
        else
          if (var4 == var5)
        {
          var6[var4][var5] = var6[var4][var5] + var6[var4 - 1][var5 - 1];
        }
        else
        {
          if (var6[var4 - 1][var5 - 1] > var6[var4 - 1][var5])
          {
            var6[var4][var5] = var6[var4][var5] + var6[var4 - 1][var5 - 1];
          }
          else
          {
            var6[var4][var5] = var6[var4][var5] + var6[var4 - 1][var5];
          }
        }
      }
    }
    for (var5 = 0; var5 < var3; var5++)
    {
      if (var6[var3 - 1][var5] > var1)
      {
        var1 = var6[var3 - 1][var5];
      }
    }
    printf("%d\n", var1);
    var2 = var2 - 1;
  }
  return 0;
}
```
# code after
```c
int main(void)
{
  int test_case_count = 1;
  int max_path_sum = 0;
  scanf("%d", &test_case_count);
  while (test_case_count > 0)
  {
    int triangle_size;
    int row;
    int col;
    scanf("%d", &triangle_size);
    int triangle[triangle_size][triangle_size];

    for (row = 0; row < triangle_size; row++)
    {
      for (col = 0; col < triangle_size; col++)
      {
        triangle[row][col] = 0;
      }
    }

    for (row = 0; row < triangle_size; row++)
    {
      for (col = 0; col <= row; col++)
      {
        scanf("%d", &triangle[row][col]);
      }
    }

    for (row = 1; row < triangle_size; row++)
    {
      for (col = 0; col <= row; col++)
      {
        if (col == 0)
        {
          triangle[row][col] += triangle[row - 1][col];
        }
        else if (row == col)
        {
          triangle[row][col] += triangle[row - 1][col - 1];
        }
        else
        {
          if (triangle[row - 1][col - 1] > triangle[row - 1][col])
          {
            triangle[row][col] += triangle[row - 1][col - 1];
          }
          else
          {
            triangle[row][col] += triangle[row - 1][col];
          }
        }
      }
    }

    for (col = 0; col < triangle_size; col++)
    {
      if (triangle[triangle_size - 1][col] > max_path_sum)
      {
        max_path_sum = triangle[triangle_size - 1][col];
      }
    }

    printf("%d\n", max_path_sum);
    test_case_count--;
  }
  return 0;
}
```
According to the example above，Please process the flowing code:
"""

model_name = 'codebert'  # graphcodebert codebert  codet5
input_path = ''
output_path = ''


if not os.path.exists(output_path):
    os.makedirs(output_path)
files = os.listdir(input_path)
num = 0
sorted_files = sorted(files, key=lambda x: int(x.split("_")[0]))

for file in sorted_files:
    num += 1
    if os.path.isfile(output_path + '/' + file):
        print("continue")
        continue
    print(file)
    with open(input_path + '/' + file, 'r', encoding='utf') as f1:
        code = f1.read()
        print(code)
        ans = abstract_the_code(code, prompt_c)
        if ans == 'error1':
            print('#####-----#####')
        if ans == 'error2':
            print('%%%%%%%%%%%----%%%%%%%%%%%')
        else:
            print(ans)
            with open(output_path + '/' + file, 'w') as w:
                # w.write(ans)
                lines = ans.splitlines()
                if len(lines) > 2:
                    content = "\n".join(lines[1:-1])
                    w.write(content)



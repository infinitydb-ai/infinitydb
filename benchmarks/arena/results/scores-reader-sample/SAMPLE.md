# Local reader sample (generated)

model `llama3.2:3b` via Ollama, CPU only; conv-26, first 22 questions per system. Real generations, no projection.

| system | questions answered | F1 | gold string in answer |
|---|---|---|---|
| infinitydb | 14 | 0.433 | 0.286 |
| infinitydb-dense | 14 | 0.437 | 0.214 |
| mem0 | 18 | 0.381 | 0.111 |

## Verbatim answers

```
Q: When did Caroline go to the LGBTQ support group?
  gold   : 7 May 2023
  infinitydb      : '08 May 2023'   F1=0.67
Q: When did Melanie paint a sunrise?
  gold   : 2022
  infinitydb      : 'Last year.'   F1=0.00
Q: When did Caroline go to the LGBTQ support group?
  gold   : 7 May 2023
  infinitydb-dense: '08 May 2023'   F1=0.67
Q: When did Melanie paint a sunrise?
  gold   : 2022
  infinitydb-dense: 'Last year.'   F1=0.00
Q: When did Caroline go to the LGBTQ support group?
  gold   : 7 May 2023
  mem0            : '2023-05-08'   F1=0.33
Q: When did Melanie paint a sunrise?
  gold   : 2022
  mem0            : '[2023-05-08]'   F1=0.00
```

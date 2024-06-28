1.由于 $\bar A,\bar B$ 都是递归可枚举的。

$$
\exists T_A,s.t. \forall x\in \bar A,T_A \textbf{halts on }x\\
\exists T_B,s.t. \forall x\in \bar B,T_B \textbf{halts on }x
$$

令 $L=\bar B$，由于 $A\cap B=\empty,A\subseteq L,\bar L \subseteq \bar A$，取通用图灵机 $U$，同步模拟 $T_A,T_B$，则对于任意输入 $x$，有：

若 $x\in L=\bar B$，$T_B$ 停机而 $T_A$ 不停机。
若 $x\in \bar L \subseteq \bar A$，$T_A$ 停机而 $T_B$ 不停机。

故 $U$ 必停机，则 $A,B$ 是递归可区分的。

2.直接 tarjan 或者 kosaraju 缩 SCC 成 DAG 即可，细节省略。
一般来说这东西可以改一改，变成最大权闭合子图。

3.只需要构造随机数生成器 $R(i)$ 使其生成 $1-i$ 中的均匀分布即可，期望复杂度为 $O(\log i)$，显然我们只需要调用抛硬币 $t=\lceil \log_2 i\rceil$ 次，如果结果不小于 $i$ 就重新抛直到结果小于 $i$ 即可。由于失败概率不超过 $1/2$，显然重复次数为 $O(1)$，最终期望复杂度 $O(\log i)$。

剩下的就是Knuth洗牌，细节省略。


4.$$v_0(y_0,y_1)=-v_1(y_0,y_1)\leq-v_1(y_0,x_1)=v_0(y_0,x_1)<u_0(y_0,x_1)\leq u_0(x_0,x_1)$$

这就是 零和博弈特殊性，纳什均衡之下，你乱动会导致对手收益上升。

5. 奇数项偶数项分别夹逼即可。

设 $t=[b_0,b_1,\dots]$。

对于奇数有：

$$\lim_{n\to\infty}[a_0,\dots,a_{2n}]\leq \lim_{n\to \infty}[a_0,\dots,a_{2n+1}+t]\leq \lim_{n\to \infty}[a_0,\dots,a_{2n+1}]$$

对于偶数有：

$$
\lim_{n\to \infty}[a_0,\dots a_{2n-1}]\geq\lim_{n\to \infty}[a_0,\dots,a_{2n}+t]\geq lim_{n\to \infty}[a_0,\dots,a_{2n}]
$$

搞定。
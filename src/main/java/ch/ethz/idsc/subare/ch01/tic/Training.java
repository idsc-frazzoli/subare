// code by jph
package ch.ethz.idsc.subare.ch01.tic;

class Training {
  public static void train(int epochs) {
    Agent player1 = new Agent(1);
    player1.setRates(0.15, 0.1);
    // player1.estimation.init(player1.symbol);
    Agent player2 = new Agent(-1);
    // player2.estimation.init(player2.symbol);
    player1.setRates(0.15, 0.1);
    Judger judger = new Judger(player1, player2, true);
    double player1Win = 0.0;
    double player2Win = 0.0;
    for (int i = 0; i < epochs; ++i) {
      final int winner = judger.play(false).winner;
      if (winner == 1)
        player1Win += 1;
      if (winner == -1)
        player2Win += 1;
    }
    player1Win /= epochs;
    player2Win /= epochs;
    System.out.println(player1Win);
    System.out.println(player2Win);
    player1.savePolicy();
    player2.savePolicy();
  }

  static void play() {
    while (true) {
      Agent player1 = new Agent(1);
      player1.loadPolicy();
      Agent player2 = new HumanAgent(-1);
      Judger judger = new Judger(player1, player2, true);
      State state = judger.play(true);
      System.out.println("final state:\n" + state);
    }
  }

  public static void main(String[] args) {
    train(20000);
    // Analysis.showVariance();
    {
      Estimation estimation = Estimation.load(1);
      System.out.println("P_win(   0)=" + //
          estimation.get(AllStates.INSTANCE.getFromHash(0)));
      System.out.println("P_win(8338)=" + //
          estimation.get(AllStates.INSTANCE.getFromHash(8338)));
    }
    play();
    // System.out.println(BinomialCoefficient.numpyRandomBinomial(1, 0.0));
  }
}

// code by jph
// adapted from code by Shangtong Zhang
package ch.ethz.idsc.subare.ch01.tic;

class Judger {
  static final int p1Symbol = 1;
  static final int p2Symbol = -1;
  // ---
  final Agent p1;
  final Agent p2;
  final boolean feedback;
  State currentState;
  Agent currentPlayer;

  Judger(Agent ply1, Agent ply2, boolean feedback) {
    p1 = ply1;
    p2 = ply2;
    this.feedback = feedback;
    // GlobalAssert.of(p1.symbol == p1Symbol);
    // GlobalAssert.of(p2.symbol == p2Symbol);
  }

  void reset() {
    p1.reset();
    p2.reset();
    currentState = State.empty;
    currentPlayer = null;
  }

  void giveReward() {
    if (currentState.winner == p1Symbol) {
      p1.feedReward(1);
      p2.feedReward(0);
    } else //
    if (currentState.winner == p2Symbol) {
      p1.feedReward(0);
      p2.feedReward(1);
    } else {
      p1.feedReward(0);
      p2.feedReward(0);
    }
  }

  private void feedCurrentState() {
    p1.feedState(currentState);
    p2.feedState(currentState);
  }

  public State play(boolean show) {
    reset();
    feedCurrentState();
    // append state of empty board
    while (true) {
      // toggle current player
      currentPlayer = currentPlayer == p1 ? p2 : p1;
      // ---
      if (show) {
        System.out.println("-------");
        System.out.println(currentState);
      }
      // ---
      Action action = currentPlayer.takeAction();
      currentState = currentState.nextState(action.pos, action.symbol);
      feedCurrentState();
      if (currentState.isEnd()) {
        if (feedback)
          giveReward();
        return currentState;
      }
    }
  }
}

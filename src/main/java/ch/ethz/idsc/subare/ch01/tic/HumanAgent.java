// code by jph
package ch.ethz.idsc.subare.ch01.tic;

import java.util.Scanner;

/* package */ class HumanAgent extends Agent {
  HumanAgent(int symbol) {
    super(symbol);
  }

  @Override
  public Action takeAction() {
    try (Scanner scanner = new Scanner(System.in)) {
      int pos;
      System.out.println("Enter pos:");
      pos = scanner.nextInt();
      if (currentState.data[pos] != 0)
        return takeAction(); // another try
      return new Action(pos, symbol);
    }
  }
}

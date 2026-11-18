// Programmable arithmetic kernel only: memory and all EEG stages are external.
// Accept one signed x,w pair on every in_valid rising edge. Twenty-two accepted
// pairs form a dot product. result is meaningful only while out_valid is high.
module dot22 #(
    parameter integer WB = 12,
    parameter integer AB = 16 + WB + 4
) (
    input wire clk,
    input wire rst_n,
    input wire in_valid,
    input wire signed [15:0] x,
    input wire signed [WB-1:0] w,
    output reg out_valid,
    output wire signed [AB-1:0] result
);
    localparam integer PB = 16 + WB;
    reg [4:0] count;
    reg signed [AB-1:0] acc;
    wire signed [PB-1:0] product = $signed(x) * $signed(w);
    wire signed [AB-1:0] product_ext = {{(AB-PB){product[PB-1]}}, product};
    assign result = acc;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= 5'd0;
            acc <= {AB{1'b0}};
            out_valid <= 1'b0;
        end else begin
            out_valid <= 1'b0;
            if (in_valid) begin
                if (count == 5'd0)
                    acc <= product_ext;
                else
                    acc <= acc + product_ext;
                if (count == 5'd21) begin
                    count <= 5'd0;
                    out_valid <= 1'b1;
                end else begin
                    count <= count + 5'd1;
                end
            end
        end
    end
endmodule
